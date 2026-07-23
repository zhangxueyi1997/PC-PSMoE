# -*- coding: utf-8 -*-
"""
Fair deep-sequence baselines for the PC-PSMoE re-validation.

Motivation
----------
The tree baselines (LightGBM/XGBoost/CatBoost/ExtraTrees) receive the FULL flattened
120x8 curve, but a tree treats each timestep-channel cell as an independent feature and
cannot exploit temporal structure (convolution / recurrence / attention). So beating a
flatten-tree does NOT prove the PC-PSMoE *architecture* is better -- it mostly proves the
curve is informative (which the ablation already shows).

This module adds genuinely fair comparators: deep sequence models that consume the same
120x8 curve and the same static features, but WITHOUT PC-PSMoE's special machinery
(no Mixture-of-Experts routing, no multiplicative PKN anchor, no physics losses, no
group-DRO). They are plain regressors trained with SmoothL1 on standardized log-targets
and early stopping on the validation fold -- the standard strong deep-learning baseline.

Models exposed
--------------
  CNN1D_static            multi-scale Conv1d encoder  + static MLP
  TCN_static              dilated temporal conv (TCN) + static MLP
  LSTM_static             bidirectional LSTM          + static MLP
  Transformer_static      SAME encoder as PC-PSMoE    + static MLP   (static+seq, NO PKN)
  Transformer_seq_only    SAME encoder as PC-PSMoE    (sequence only, no static, no PKN)
  PKN_residual_LightGBM   LightGBM predicting log-residual on top of fold-calibrated PKN
  PKN_residual_XGBoost    XGBoost  predicting log-residual on top of fold-calibrated PKN

The `pc` module (train_pc_psmoe_full.py) is passed in so we reuse its FoldPreprocessor,
ResidualBlock and TemporalCurveEncoder for an apples-to-apples comparison.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None


# =============================================================================
# Sequence encoders (all map B x T x C  ->  B x width)
# =============================================================================
class LSTMEncoder(nn.Module):
    def __init__(self, n_channels: int, width: int, dropout: float) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            n_channels, width, num_layers=2, batch_first=True,
            bidirectional=True, dropout=dropout,
        )
        self.proj = nn.Sequential(nn.Linear(2 * width, width), nn.LayerNorm(width))

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(seq)            # B, T, 2*width
        pooled = out.mean(dim=1)          # mean over time
        return self.proj(pooled)


class _TCNBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = dilation  # keeps length for kernel_size=3
        self.conv1 = nn.Conv1d(width, width, 3, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, 3, padding=pad, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.gelu(self.conv1(x))
        y = self.drop(y)
        y = F.gelu(self.conv2(y))
        return self.norm(x + y)


class TCNEncoder(nn.Module):
    def __init__(self, n_channels: int, width: int, dropout: float) -> None:
        super().__init__()
        self.inp = nn.Conv1d(n_channels, width, 1)
        self.blocks = nn.Sequential(
            *[_TCNBlock(width, d, dropout) for d in (1, 2, 4, 8)]
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = seq.transpose(1, 2)           # B, C, T
        x = self.inp(x)
        x = self.blocks(x)                # B, width, T
        return self.norm(x.mean(dim=2))


class CNNEncoder(nn.Module):
    def __init__(self, n_channels: int, width: int, dropout: float) -> None:
        super().__init__()
        self.inp = nn.Conv1d(n_channels, width, 1)
        self.c3 = nn.Conv1d(width, width, 3, padding=1)
        self.c5 = nn.Conv1d(width, width, 5, padding=2)
        self.c7 = nn.Conv1d(width, width, 7, padding=3)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(width)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = seq.transpose(1, 2)
        x = self.inp(x)
        x = F.gelu(self.c3(x)) + F.gelu(self.c5(x)) + F.gelu(self.c7(x))
        x = self.drop(x)
        return self.norm(x.mean(dim=2))


class TransformerEncoder(nn.Module):
    """Reuses PC-PSMoE's exact curve encoder for the fairest architectural test."""

    def __init__(self, pc, n_channels: int, width: int, layers: int,
                 heads: int, dropout: float) -> None:
        super().__init__()
        self.enc = pc.TemporalCurveEncoder(n_channels, width, layers, heads, dropout)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.enc(seq)
        return pooled


def _build_seq_encoder(pc, kind: str, width: int, mc, dropout: float) -> nn.Module:
    if kind == "transformer":
        return TransformerEncoder(pc, 8, width, mc.transformer_layers,
                                  mc.transformer_heads, dropout)
    if kind == "lstm":
        return LSTMEncoder(8, width, dropout)
    if kind == "tcn":
        return TCNEncoder(8, width, dropout)
    if kind == "cnn":
        return CNNEncoder(8, width, dropout)
    raise ValueError(f"unknown seq encoder kind: {kind}")


class DeepRegressor(nn.Module):
    """Plain deep regressor: [seq encoder] (+ optional static MLP) -> fusion -> 3 outputs."""

    def __init__(self, pc, kind: str, static_dim: int, use_static: bool, mc,
                 width: int = 96, fusion: int = 192, dropout: float = 0.12) -> None:
        super().__init__()
        self.use_static = use_static
        self.seq = _build_seq_encoder(pc, kind, width, mc, dropout)
        in_dim = width
        if use_static:
            self.static = nn.Sequential(
                nn.Linear(static_dim, width), nn.GELU(), nn.Dropout(dropout),
                pc.ResidualBlock(width, dropout), nn.LayerNorm(width),
            )
            in_dim += width
        self.head = nn.Sequential(
            nn.Linear(in_dim, fusion), nn.GELU(), nn.Dropout(dropout),
            pc.ResidualBlock(fusion, dropout), nn.LayerNorm(fusion),
            nn.Linear(fusion, 3),
        )

    def forward(self, static: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        h = self.seq(seq)
        if self.use_static:
            h = torch.cat([h, self.static(static)], dim=1)
        return self.head(h)


# =============================================================================
# Training (standardized log-targets, SmoothL1, AdamW + cosine, early stopping)
# =============================================================================
def train_deep_baseline(pc, kind, use_static, raw, train_idx, val_idx, test_idx,
                        device, epochs, seed, patience=30, batch_size=128):
    pc.seed_everything(seed)
    pre = pc.FoldPreprocessor(
        pc.AblationConfig(variant="fair", use_static=use_static, use_curve=True,
                          use_pkn=False, use_physics_losses=False)
    )
    data = pre.fit_transform(raw, train_idx)               # fitted on train only
    static = torch.from_numpy(data.static).float().to(device)
    seq = torch.from_numpy(data.sequence).float().to(device)

    logy = np.log(np.maximum(raw.y, 1e-5)).astype(np.float64)
    m = logy[train_idx].mean(axis=0, keepdims=True)
    s = np.maximum(logy[train_idx].std(axis=0, keepdims=True), 1e-6)
    z = torch.from_numpy(((logy - m) / s).astype(np.float32)).to(device)

    mc = pc.ModelConfig()
    model = DeepRegressor(pc, kind, data.static.shape[1], use_static, mc).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=5e-6)

    tr = np.asarray(train_idx)
    va = torch.as_tensor(np.asarray(val_idx), device=device, dtype=torch.long)
    rng = np.random.default_rng(seed)

    best_state, best_val, left = None, math.inf, patience
    for _ in range(epochs):
        model.train()
        perm = tr[rng.permutation(len(tr))]
        for i in range(0, len(perm), batch_size):
            b = torch.as_tensor(perm[i:i + batch_size], device=device, dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            loss = F.smooth_l1_loss(model(static[b], seq[b]), z[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vloss = F.smooth_l1_loss(model(static[va], seq[va]), z[va]).item()
        if vloss < best_val - 1e-5:
            best_val, best_state, left = vloss, copy.deepcopy(model.state_dict()), patience
        else:
            left -= 1
            if left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tt = torch.as_tensor(np.asarray(test_idx), device=device, dtype=torch.long)
        z_hat = model(static[tt], seq[tt]).cpu().numpy()
    return np.exp(z_hat * s + m)


# =============================================================================
# PKN-residual trees (same multiplicative-PKN-anchor inductive bias as PC-PSMoE)
# =============================================================================
def pkn_residual_tree(pc, raw, fit_idx, test_idx, seed, kind):
    pre = pc.FoldPreprocessor(
        pc.AblationConfig(variant="pknres", use_static=True, use_curve=True, use_pkn=True)
    )
    data = pre.fit_transform(raw, fit_idx)                 # fitted on train+val
    pkn = data.pkn.astype(np.float64)
    feats = np.hstack(
        [data.static, data.sequence.reshape(len(data.y), -1)]
    ).astype(np.float32)
    log_pkn = np.log(np.maximum(pkn, 1e-5))
    log_resid = np.log(np.maximum(raw.y, 1e-5)) - log_pkn  # learn the correction

    out = np.zeros((len(test_idx), 3), dtype=float)
    for ti in range(3):
        if kind == "lgb":
            if LGBMRegressor is None:
                return None
            model = LGBMRegressor(
                n_estimators=180, learning_rate=0.04, num_leaves=31, max_depth=-1,
                min_child_samples=16, subsample=0.9, colsample_bytree=0.82,
                reg_alpha=0.02, reg_lambda=4.0, random_state=seed + 211 * ti,
                n_jobs=4, verbose=-1)
        else:
            if XGBRegressor is None:
                return None
            model = XGBRegressor(
                n_estimators=150, max_depth=3, learning_rate=0.045, subsample=0.88,
                colsample_bytree=0.78, min_child_weight=2.0, reg_alpha=0.02,
                reg_lambda=4.5, objective="reg:squarederror", tree_method="hist",
                random_state=seed + 101 * ti, n_jobs=4)
        model.fit(feats[fit_idx], log_resid[fit_idx, ti])
        out[:, ti] = np.exp(log_pkn[test_idx, ti] + model.predict(feats[test_idx]))
    return out


# =============================================================================
# Orchestration with per-(fold, model) caching for resumability
# =============================================================================
DEEP_SPECS = {  # name -> (kind, use_static)
    "CNN1D_static": ("cnn", True),
    "TCN_static": ("tcn", True),
    "LSTM_static": ("lstm", True),
    "Transformer_static": ("transformer", True),
    "Transformer_seq_only": ("transformer", False),
}
TREE_SPECS = {  # name -> kind
    "PKN_residual_LightGBM": "lgb",
    "PKN_residual_XGBoost": "xgb",
}
ALL_FAIR_MODELS = list(DEEP_SPECS) + list(TREE_SPECS)


def fair_baseline_predictions(pc, raw, train_idx, val_idx, test_idx, *,
                              cache_dir, fold_name, epochs, seed, device,
                              models=None, log=print):
    """Return {model_name: test_predictions}. Caches each model's test preds to disk."""
    cache = Path(cache_dir) / fold_name
    cache.mkdir(parents=True, exist_ok=True)
    fit_idx = np.sort(np.r_[np.asarray(train_idx), np.asarray(val_idx)])
    wanted = models or ALL_FAIR_MODELS
    preds: dict[str, np.ndarray] = {}
    for i, name in enumerate(wanted):
        cache_file = cache / f"{name}.npy"
        if cache_file.exists():
            preds[name] = np.load(cache_file)
            log(f"    [fair] {name}: cached")
            continue
        import time as _t
        t0 = _t.time()
        if name in DEEP_SPECS:
            kind, use_static = DEEP_SPECS[name]
            yhat = train_deep_baseline(
                pc, kind, use_static, raw, train_idx, val_idx, test_idx,
                device, epochs, seed=seed + 1000 * i)
        elif name in TREE_SPECS:
            yhat = pkn_residual_tree(pc, raw, fit_idx, test_idx,
                                     seed=seed + 1000 * i, kind=TREE_SPECS[name])
            if yhat is None:
                log(f"    [fair] {name}: SKIPPED (library not installed)")
                continue
        else:
            raise ValueError(f"unknown fair model: {name}")
        np.save(cache_file, yhat)
        preds[name] = yhat
        log(f"    [fair] {name}: {_t.time() - t0:.0f}s")
    return preds
