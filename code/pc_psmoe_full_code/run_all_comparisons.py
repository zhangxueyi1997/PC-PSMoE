# -*- coding: utf-8 -*-
"""
====================================================================================
 ONE-FILE, ONE-COMMAND full model comparison on the 55-well / 1967-stage dataset
====================================================================================

Runs EVERYTHING and prints ONE final comparison table:

  * PC-PSMoE (the proposed model)                              -> imported from train_pc_psmoe_full.py
  * Standard tree / linear baselines on the full flattened 120x8 curve + static + log-PKN
        LightGBM, XGBoost, CatBoost, ExtraTrees, RandomForest, Ridge, Calibrated-PKN,
        and ExtraTrees feature-source ablations (NoPKN / NoCurve / NoStatic)
  * FAIR deep-sequence baselines (same 120x8 curve + static, NO MoE / NO PKN anchor /
    NO physics losses / NO group-DRO):
        CNN1D_static, TCN_static, LSTM_static, Transformer_static, Transformer_seq_only
  * FAIR PKN-residual trees (tree gets the same multiplicative-PKN anchor as PC-PSMoE):
        PKN_residual_LightGBM, PKN_residual_XGBoost
  * Architecture ablations of PC-PSMoE itself (paper Table 7)

Protocols: random hold-out (60/20/20) and leave-one-well-out (LOWO, one fold per well),
plus Bootstrap 95% CIs and per-sample paired Wilcoxon / t tests. All preprocessing, PKN
calibration, model selection and conformal calibration are fitted strictly inside each
fold (no leakage).

------------------------------------------------------------------------------------
 HOW TO RUN
------------------------------------------------------------------------------------
 Keep `train_pc_psmoe_full.py` (the PC-PSMoE model) in the SAME folder as this file.
 Install once:
     pip install torch numpy pandas scikit-learn scipy joblib openpyxl xgboost lightgbm catboost

 Commands:
     python run_all_comparisons.py                       # EVERYTHING (random + LOWO, ~ several h CPU)
     python run_all_comparisons.py --protocols random    # random hold-out only (fast & decisive, ~40 min)
     python run_all_comparisons.py --quick               # 8-epoch wiring smoke test (~6 min)
     python run_all_comparisons.py --no-fair             # only PC-PSMoE + standard trees (paper-style)
     python run_all_comparisons.py --device cuda         # use a GPU if available

 RESUMABLE: finished PC folds and cached baseline predictions are reused on restart.
 On Windows it asks the OS to stay awake while running.

------------------------------------------------------------------------------------
 MAIN OUTPUTS (in --output-dir, default 03_validation_results/all_comparisons_18wells)
------------------------------------------------------------------------------------
   final_comparison_table.csv   <-- THE single consolidated table (all models, both protocols)
   comparison_to_paper.txt      full report incl. ablations, fair-baseline verdict, paired tests
   all_model_fold_metrics.csv   per-fold metrics for every model
   *_summary.csv, *_bootstrap_ci.csv, *_paired_tests.csv
   all_comparisons_18wells.xlsx everything in one workbook
====================================================================================
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scipy import stats
except Exception:  # pragma: no cover
    print("scipy not found")
    stats = None
try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    print("scipy not found")
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    print("scipy not found")
    LGBMRegressor = None
try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover
    print("scipy not found")
    CatBoostRegressor = None

SCRIPT_DIR = Path(__file__).resolve().parent
TARGETS = ["L_m", "W_m", "H_m"]
TARGET_PREFIX = {"L_m": "L", "W_m": "W", "H_m": "H"}
ABLATION_VARIANTS = ["full", "no_pkn", "no_curve", "no_static",
                     "no_physics", "no_group_dro", "single_expert"]

# fair-baseline registries
DEEP_SPECS = {  # name -> (encoder_kind, use_static)
    "CNN1D_static": ("cnn", True),
    "TCN_static": ("tcn", True),
    "LSTM_static": ("lstm", True),
    "Transformer_static": ("transformer", True),
    "Transformer_seq_only": ("transformer", False),
}
TREE_SPECS = {"PKN_residual_LightGBM": "lgb", "PKN_residual_XGBoost": "xgb"}
ALL_FAIR_MODELS = list(DEEP_SPECS) + list(TREE_SPECS)
FAIR_DEEP = list(DEEP_SPECS)
FAIR_TREE = list(TREE_SPECS)
STANDARD_BASELINES = ["LightGBM_AllFeatures", "XGBoost_AllFeatures",
                      "CatBoost_AllFeatures", "Ridge_AllFeatures",
                      "ExtraTrees_AllFeatures", "RandomForest_AllFeatures",
                      "Calibrated_PKN"]
CATEGORY = {"PC-PSMoE": "proposed"}
for _m in FAIR_DEEP:
    CATEGORY[_m] = "fair deep (curve+static, no PKN/MoE)"
for _m in FAIR_TREE:
    CATEGORY[_m] = "fair tree (PKN-residual anchor)"
for _m in ["LightGBM_AllFeatures", "XGBoost_AllFeatures", "CatBoost_AllFeatures",
           "ExtraTrees_AllFeatures", "RandomForest_AllFeatures"]:
    CATEGORY[_m] = "tree (flatten curve)"
CATEGORY["Ridge_AllFeatures"] = "linear (flatten curve)"
CATEGORY["Calibrated_PKN"] = "physics prior"
for _m in ["ExtraTrees_NoPKN", "ExtraTrees_NoCurve", "ExtraTrees_NoStatic"]:
    CATEGORY[_m] = "tree feature-ablation"

# ---- Paper-reported numbers (OLD dataset), for the comparison ----------------
PAPER_RANDOM_TARGET = {"L_m": (0.917, 16.49), "W_m": (0.958, 3.47),
                       "H_m": (0.947, 1.92), "mean": (0.941, 7.29)}
PAPER_BASELINES = {  # model -> (rnd_R2, rnd_RMSE, LOWO_R2, LOWO_RMSE)
    "PC-PSMoE": (0.941, 7.29, 0.926, 8.16),
    "LightGBM_AllFeatures": (0.884, 7.58, 0.879, 7.81),
    "XGBoost_AllFeatures": (0.865, 8.38, 0.858, 8.79),
    "CatBoost_AllFeatures": (0.862, 8.77, 0.860, 8.85),
    "Ridge_AllFeatures": (0.817, 10.28, 0.797, 11.02),
    "ExtraTrees_AllFeatures": (0.804, 9.05, 0.773, 9.95),
    "RandomForest_AllFeatures": (0.783, 9.56, 0.754, 10.25),
    "Calibrated_PKN": (0.241, 23.42, 0.237, 23.28),
}
PAPER_ABLATION = {"full": (0.951, 0.000), "no_pkn": (0.945, -0.006),
                  "no_group_dro": (0.942, -0.009), "no_physics": (0.938, -0.013),
                  "single_expert": (0.934, -0.017), "no_static": (0.839, -0.112),
                  "no_curve": (0.426, -0.525)}
PAPER_LOWO_MACRO = {"L_m": (0.885, 0.045, 0.896), "W_m": (0.932, 0.021, 0.942),
                    "H_m": (0.936, 0.024, 0.939), "mean": (0.917, 0.027, 0.926)}


# =============================================================================
# Environment helpers
# =============================================================================
def prevent_sleep() -> None:
    try:
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception as exc:  # pragma: no cover
        print(f"prevent_sleep failed: {exc}", flush=True)


def find_project_root() -> Path:
    for base in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        if (base / "data" / "completion_geomechanics_fracture_parameters.xlsx").exists():
            return base
        staged = (base / "data" / "field_records"
                  / "completion_geomechanics_parameters.xlsx")
        if staged.exists():
            return base
        staged = (base / "01_final_data" / "field_records"
                  / "completion_geomechanics_parameters.xlsx")
        if staged.exists():
            return base
    return SCRIPT_DIR.parents[1] if len(SCRIPT_DIR.parents) >= 2 else SCRIPT_DIR


def find_model_script() -> Path:
    for cand in [SCRIPT_DIR / "train_pc_psmoe_full.py",
                 SCRIPT_DIR / "train_full_architecture.py",
                 Path.cwd() / "train_pc_psmoe_full.py"]:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "Could not find train_pc_psmoe_full.py (the PC-PSMoE model). "
        "Put it in the SAME folder as this script."
    )


def ensure_data(project_root: Path) -> Path:
    expected_dir = next(
        (p for p in [
            project_root / "data" / "field_records",
            project_root / "01_final_data" / "field_records",
        ] if p.exists()),
        project_root / "data" / "field_records",
    )
    xlsx = expected_dir / "completion_geomechanics_parameters.xlsx"
    csv = expected_dir / "treatment_curve_matrix_120step.csv"
    if xlsx.exists() and csv.exists():
        return expected_dir
    src_xlsx = project_root / "data" / "completion_geomechanics_fracture_parameters.xlsx"
    src_csv = project_root / "data" / "treatment_curve_matrix_120step.csv"
    if not (src_xlsx.exists() and src_csv.exists()):
        raise FileNotFoundError(
            f"Dataset not found. Expected staged files in {expected_dir} "
            f"or source files {src_xlsx} and {src_csv}."
        )
    expected_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_xlsx, xlsx)
    shutil.copy2(src_csv, csv)
    return expected_dir


def load_model_module():
    path = find_model_script()
    spec = importlib.util.spec_from_file_location("pc_psmoe_model", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# =============================================================================
# Metrics + standard tree/linear baselines + statistics
# =============================================================================
def metric_row(protocol, fold, model, y_true, pred) -> dict[str, Any]:
    r2 = r2_score(y_true, pred, multioutput="raw_values")
    rmse = np.sqrt(mean_squared_error(y_true, pred, multioutput="raw_values"))
    mae = mean_absolute_error(y_true, pred, multioutput="raw_values")
    row: dict[str, Any] = {
        "protocol": protocol, "fold": fold, "model": model, "n_test": int(len(y_true)),
        "mean_r2": float(np.mean(r2)), "mean_rmse": float(np.mean(rmse)),
        "mean_mae": float(np.mean(mae)),
    }
    for i, t in enumerate(TARGETS):
        p = TARGET_PREFIX[t]
        row[f"{p}_r2"], row[f"{p}_rmse"], row[f"{p}_mae"] = float(r2[i]), float(rmse[i]), float(mae[i])
    return row


def make_features(pc, raw, train_idx, blocks):
    use_static, use_curve, use_pkn = "static" in blocks, "curve" in blocks, "pkn" in blocks
    pre = pc.FoldPreprocessor(pc.AblationConfig(
        variant=f"baseline_{blocks}", use_static=use_static, use_curve=use_curve,
        use_pkn=use_pkn, use_physics_losses=True))
    prepared = pre.fit_transform(raw, train_idx)
    parts = []
    if use_static:
        parts.append(prepared.static)
    if use_curve:
        parts.append(prepared.sequence.reshape(len(prepared.y), -1))
    if use_pkn:
        parts.append(np.log(np.maximum(prepared.pkn, 1e-5)))
    if not parts:
        raise ValueError("At least one feature block is required.")
    return np.hstack(parts).astype(np.float32), prepared.pkn.astype(np.float32)


def fit_predict_targetwise(factory, x, y, fit_idx, test_idx):
    out = np.zeros((len(test_idx), y.shape[1]), dtype=float)
    for ti in range(y.shape[1]):
        model = factory(ti)
        model.fit(x[fit_idx], y[fit_idx, ti])
        out[:, ti] = model.predict(x[test_idx])
    return out


def strong_baseline_predictions(pc, raw, fit_idx, test_idx, seed):
    y = raw.y.astype(float)
    x_all, pkn = make_features(pc, raw, fit_idx, "static_curve_pkn")
    x_no_pkn, _ = make_features(pc, raw, fit_idx, "static_curve")
    x_static, _ = make_features(pc, raw, fit_idx, "static_pkn")
    x_curve, _ = make_features(pc, raw, fit_idx, "curve_pkn")

    preds: dict[str, np.ndarray] = {"Calibrated_PKN": pkn[test_idx]}
    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 16)))
    ridge.fit(x_all[fit_idx], y[fit_idx])
    preds["Ridge_AllFeatures"] = ridge.predict(x_all[test_idx])

    et = ExtraTreesRegressor(n_estimators=240, min_samples_leaf=2, max_features=0.75,
                             random_state=seed, n_jobs=-1)
    et.fit(x_all[fit_idx], y[fit_idx]); preds["ExtraTrees_AllFeatures"] = et.predict(x_all[test_idx])
    rf = RandomForestRegressor(n_estimators=220, min_samples_leaf=3, max_features=0.65,
                               random_state=seed + 11, n_jobs=-1)
    rf.fit(x_all[fit_idx], y[fit_idx]); preds["RandomForest_AllFeatures"] = rf.predict(x_all[test_idx])

    if XGBRegressor is not None:
        preds["XGBoost_AllFeatures"] = fit_predict_targetwise(
            lambda ti: XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.045,
                subsample=0.88, colsample_bytree=0.78, min_child_weight=2.0, reg_alpha=0.02,
                reg_lambda=4.5, objective="reg:squarederror", tree_method="hist",
                random_state=seed + 101 * ti, n_jobs=4), x_all, y, fit_idx, test_idx)
    if LGBMRegressor is not None:
        preds["LightGBM_AllFeatures"] = fit_predict_targetwise(
            lambda ti: LGBMRegressor(n_estimators=180, learning_rate=0.04, num_leaves=31,
                max_depth=-1, min_child_samples=16, subsample=0.9, colsample_bytree=0.82,
                reg_alpha=0.02, reg_lambda=4.0, random_state=seed + 211 * ti, n_jobs=4,
                verbose=-1), x_all, y, fit_idx, test_idx)
    if CatBoostRegressor is not None:
        preds["CatBoost_AllFeatures"] = fit_predict_targetwise(
            lambda ti: CatBoostRegressor(iterations=150, depth=5, learning_rate=0.045,
                l2_leaf_reg=6.0, loss_function="RMSE", random_seed=seed + 307 * ti,
                verbose=False, allow_writing_files=False, thread_count=4),
            x_all, y, fit_idx, test_idx)

    for nm, xb, mf, sd in [("ExtraTrees_NoPKN", x_no_pkn, 0.75, seed + 401),
                           ("ExtraTrees_NoCurve", x_static, 0.85, seed + 503),
                           ("ExtraTrees_NoStatic", x_curve, 0.65, seed + 607)]:
        m = ExtraTreesRegressor(n_estimators=220, min_samples_leaf=2, max_features=mf,
                                random_state=sd, n_jobs=-1)
        m.fit(xb[fit_idx], y[fit_idx]); preds[nm] = m.predict(xb[test_idx])
    return preds


def bootstrap_ci(y_true, pred, n_boot=2000, seed=20260611):
    rng = np.random.default_rng(seed)
    targets = TARGETS + ["mean_LWH"]
    scores = {t: [] for t in targets}
    for _ in range(n_boot):
        s = rng.integers(0, len(y_true), len(y_true))
        r2 = r2_score(y_true[s], pred[s], multioutput="raw_values")
        rmse = np.sqrt(mean_squared_error(y_true[s], pred[s], multioutput="raw_values"))
        mae = mean_absolute_error(y_true[s], pred[s], multioutput="raw_values")
        for i, t in enumerate(TARGETS):
            scores[t].append((r2[i], rmse[i], mae[i]))
        scores["mean_LWH"].append((float(np.mean(r2)), float(np.mean(rmse)), float(np.mean(mae))))
    rows = []
    for t in targets:
        a = np.asarray(scores[t], dtype=float)
        rows.append({"target": t, "R2_ci2.5": float(np.percentile(a[:, 0], 2.5)),
                     "R2_ci97.5": float(np.percentile(a[:, 0], 97.5)),
                     "RMSE_ci2.5": float(np.percentile(a[:, 1], 2.5)),
                     "RMSE_ci97.5": float(np.percentile(a[:, 1], 97.5)),
                     "MAE_ci2.5": float(np.percentile(a[:, 2], 2.5)),
                     "MAE_ci97.5": float(np.percentile(a[:, 2], 97.5))})
    return pd.DataFrame(rows)


def paired_tests(protocol, pc_name, y_true, pc_pred, competitors):
    scale = np.std(y_true, axis=0, ddof=1)
    scale = np.where(scale < 1e-8, 1.0, scale)
    pc_err = np.mean(np.abs(y_true - pc_pred) / scale, axis=1)
    rows = []
    for model, pred in competitors.items():
        diff = np.mean(np.abs(y_true - pred) / scale, axis=1) - pc_err
        row = {"protocol": protocol, "pc_model": pc_name, "competitor": model,
               "n": int(len(diff)), "mean_scaled_abs_error_gain": float(diff.mean()),
               "median_scaled_abs_error_gain": float(np.median(diff)),
               "pc_better_fraction": float((diff > 0).mean())}
        if stats is not None:
            row["wilcoxon_p_less_error"] = (float(stats.wilcoxon(diff, alternative="greater").pvalue)
                                            if np.any(np.abs(diff) > 1e-12) else 1.0)
            row["paired_t_p_less_error"] = float(stats.ttest_1samp(diff, 0.0, alternative="greater").pvalue)
        else:
            row["wilcoxon_p_less_error"] = row["paired_t_p_less_error"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_model_rows(rows, protocol):
    sub = rows[rows["protocol"] == protocol].copy()
    cols = [c for c in sub.columns if c not in {"protocol", "fold", "model"}
            and pd.api.types.is_numeric_dtype(sub[c])]
    out = sub.groupby("model")[cols].agg(["mean", "std"])
    out.columns = [f"{n}_{s}" for n, s in out.columns]
    return out.reset_index().sort_values("mean_r2_mean", ascending=False)


# =============================================================================
# FAIR deep-sequence baselines (same inputs, none of PC-PSMoE's machinery)
# =============================================================================
class LSTMEncoder(nn.Module):
    def __init__(self, n_channels, width, dropout):
        super().__init__()
        self.rnn = nn.LSTM(n_channels, width, num_layers=2, batch_first=True,
                           bidirectional=True, dropout=dropout)
        self.proj = nn.Sequential(nn.Linear(2 * width, width), nn.LayerNorm(width))

    def forward(self, seq):
        out, _ = self.rnn(seq)
        return self.proj(out.mean(dim=1))


class _TCNBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(width)

    def forward(self, x):
        y = self.drop(F.gelu(self.conv1(x)))
        y = F.gelu(self.conv2(y))
        return self.norm(x + y)


class TCNEncoder(nn.Module):
    def __init__(self, n_channels, width, dropout):
        super().__init__()
        self.inp = nn.Conv1d(n_channels, width, 1)
        self.blocks = nn.Sequential(*[_TCNBlock(width, d, dropout) for d in (1, 2, 4, 8)])
        self.norm = nn.LayerNorm(width)

    def forward(self, seq):
        x = self.blocks(self.inp(seq.transpose(1, 2)))
        return self.norm(x.mean(dim=2))


class CNNEncoder(nn.Module):
    def __init__(self, n_channels, width, dropout):
        super().__init__()
        self.inp = nn.Conv1d(n_channels, width, 1)
        self.c3 = nn.Conv1d(width, width, 3, padding=1)
        self.c5 = nn.Conv1d(width, width, 5, padding=2)
        self.c7 = nn.Conv1d(width, width, 7, padding=3)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(width)

    def forward(self, seq):
        x = self.inp(seq.transpose(1, 2))
        x = self.drop(F.gelu(self.c3(x)) + F.gelu(self.c5(x)) + F.gelu(self.c7(x)))
        return self.norm(x.mean(dim=2))


class TransformerEncoder(nn.Module):
    """Reuses PC-PSMoE's exact curve encoder for the fairest architecture test."""
    def __init__(self, pc, n_channels, width, layers, heads, dropout):
        super().__init__()
        self.enc = pc.TemporalCurveEncoder(n_channels, width, layers, heads, dropout)

    def forward(self, seq):
        return self.enc(seq)[0]


def _build_seq_encoder(pc, kind, width, mc, dropout):
    if kind == "transformer":
        return TransformerEncoder(pc, 8, width, mc.transformer_layers, mc.transformer_heads, dropout)
    if kind == "lstm":
        return LSTMEncoder(8, width, dropout)
    if kind == "tcn":
        return TCNEncoder(8, width, dropout)
    if kind == "cnn":
        return CNNEncoder(8, width, dropout)
    raise ValueError(kind)


class DeepRegressor(nn.Module):
    def __init__(self, pc, kind, static_dim, use_static, mc, width=96, fusion=192, dropout=0.12):
        super().__init__()
        self.use_static = use_static
        self.seq = _build_seq_encoder(pc, kind, width, mc, dropout)
        in_dim = width
        if use_static:
            self.static = nn.Sequential(
                nn.Linear(static_dim, width), nn.GELU(), nn.Dropout(dropout),
                pc.ResidualBlock(width, dropout), nn.LayerNorm(width))
            in_dim += width
        self.head = nn.Sequential(
            nn.Linear(in_dim, fusion), nn.GELU(), nn.Dropout(dropout),
            pc.ResidualBlock(fusion, dropout), nn.LayerNorm(fusion), nn.Linear(fusion, 3))

    def forward(self, static, seq):
        h = self.seq(seq)
        if self.use_static:
            h = torch.cat([h, self.static(static)], dim=1)
        return self.head(h)


def train_deep_baseline(pc, kind, use_static, raw, train_idx, val_idx, test_idx,
                        device, epochs, seed, patience=30, batch_size=128):
    pc.seed_everything(seed)
    pre = pc.FoldPreprocessor(pc.AblationConfig(
        variant="fair", use_static=use_static, use_curve=True, use_pkn=False,
        use_physics_losses=False))
    data = pre.fit_transform(raw, train_idx)
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


def pkn_residual_tree(pc, raw, fit_idx, test_idx, seed, kind):
    pre = pc.FoldPreprocessor(pc.AblationConfig(
        variant="pknres", use_static=True, use_curve=True, use_pkn=True))
    data = pre.fit_transform(raw, fit_idx)
    pkn = data.pkn.astype(np.float64)
    feats = np.hstack([data.static, data.sequence.reshape(len(data.y), -1)]).astype(np.float32)
    log_pkn = np.log(np.maximum(pkn, 1e-5))
    log_resid = np.log(np.maximum(raw.y, 1e-5)) - log_pkn
    out = np.zeros((len(test_idx), 3), dtype=float)
    for ti in range(3):
        if kind == "lgb":
            if LGBMRegressor is None:
                return None
            model = LGBMRegressor(n_estimators=180, learning_rate=0.04, num_leaves=31,
                max_depth=-1, min_child_samples=16, subsample=0.9, colsample_bytree=0.82,
                reg_alpha=0.02, reg_lambda=4.0, random_state=seed + 211 * ti, n_jobs=4, verbose=-1)
        else:
            if XGBRegressor is None:
                return None
            model = XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.045,
                subsample=0.88, colsample_bytree=0.78, min_child_weight=2.0, reg_alpha=0.02,
                reg_lambda=4.5, objective="reg:squarederror", tree_method="hist",
                random_state=seed + 101 * ti, n_jobs=4)
        model.fit(feats[fit_idx], log_resid[fit_idx, ti])
        out[:, ti] = np.exp(log_pkn[test_idx, ti] + model.predict(feats[test_idx]))
    return out


def fair_baseline_predictions(pc, raw, train_idx, val_idx, test_idx, *, cache_dir,
                              fold_name, epochs, seed, device, models=None, log=print):
    cache = Path(cache_dir) / fold_name
    cache.mkdir(parents=True, exist_ok=True)
    fit_idx = np.sort(np.r_[np.asarray(train_idx), np.asarray(val_idx)])
    wanted = models or ALL_FAIR_MODELS
    preds: dict[str, np.ndarray] = {}
    for i, name in enumerate(wanted):
        cache_file = cache / f"{name}.npy"
        if cache_file.exists():
            preds[name] = np.load(cache_file); log(f"    [fair] {name}: cached"); continue
        t0 = time.time()
        if name in DEEP_SPECS:
            kind, use_static = DEEP_SPECS[name]
            yhat = train_deep_baseline(pc, kind, use_static, raw, train_idx, val_idx,
                                       test_idx, device, epochs, seed=seed + 1000 * i)
        elif name in TREE_SPECS:
            yhat = pkn_residual_tree(pc, raw, fit_idx, test_idx, seed=seed + 1000 * i,
                                     kind=TREE_SPECS[name])
            if yhat is None:
                log(f"    [fair] {name}: SKIPPED (library missing)"); continue
        else:
            raise ValueError(name)
        np.save(cache_file, yhat); preds[name] = yhat
        log(f"    [fair] {name}: {time.time() - t0:.0f}s")
    return preds


# =============================================================================
# PC-PSMoE fold runner (resumable)
# =============================================================================
def run_pc_fold(pc, raw, split, variant, out_root, fold_number, epochs, seed, device):
    fold_dir = out_root / split.name
    manifest_path = fold_dir / "fold_manifest.json"
    if manifest_path.exists() and (fold_dir / "predictions.csv").exists():
        m = json.loads(manifest_path.read_text(encoding="utf-8"))["metrics"]
        m["_reused"] = True
        return m
    model_config = pc.ModelConfig()
    train_config = pc.TrainConfig(seed=seed, epochs=epochs)
    ablation_config = pc.apply_variant_to_configs(variant, model_config, train_config)
    fold_train_config = copy.deepcopy(train_config)
    fold_train_config.seed = seed + fold_number * 101
    out_root.mkdir(parents=True, exist_ok=True)
    m = pc.fit_one_fold(raw, split, model_config, fold_train_config, ablation_config,
                        out_root, device)
    m["_reused"] = False
    return m


def load_pc_predictions(fold_dir: Path):
    df = pd.read_csv(fold_dir / "predictions.csv")
    y_true = df[[f"true_{t}" for t in TARGETS]].to_numpy(dtype=float)
    y_pred = df[[f"pred_{t}" for t in TARGETS]].to_numpy(dtype=float)
    return y_true, y_pred


# =============================================================================
# Reporting: ONE consolidated comparison table + full report
# =============================================================================
def fmt(x, nd=3):
    try:
        v = float(x)
        return "NA" if math.isnan(v) else f"{v:.{nd}f}"
    except Exception:
        return str(x)


def build_final_table(out: Path) -> pd.DataFrame:
    def _read(name):
        p = out / name
        return pd.read_csv(p) if p.exists() else None
    rsum = _read("random_holdout_model_summary.csv")
    pooled = _read("lowo_pooled_summary.csv")
    rsum = rsum.set_index("model") if rsum is not None else None
    pooled = pooled.set_index("model") if pooled is not None else None

    models = set()
    if rsum is not None:
        models |= set(rsum.index)
    if pooled is not None:
        models |= set(pooled.index)
    rows = []
    for m in models:
        r_r2 = rsum.loc[m, "mean_r2_mean"] if (rsum is not None and m in rsum.index) else np.nan
        r_rmse = rsum.loc[m, "mean_rmse_mean"] if (rsum is not None and m in rsum.index) else np.nan
        l_r2 = pooled.loc[m, "mean_r2"] if (pooled is not None and m in pooled.index) else np.nan
        l_rmse = pooled.loc[m, "mean_rmse"] if (pooled is not None and m in pooled.index) else np.nan
        pap = PAPER_BASELINES.get(m, (np.nan,) * 4)
        rows.append({"model": m, "category": CATEGORY.get(m, "other"),
                     "random_R2": r_r2, "random_RMSE": r_rmse,
                     "LOWO_R2": l_r2, "LOWO_RMSE": l_rmse,
                     "paper_random_R2": pap[0], "paper_LOWO_R2": pap[2]})
    df = pd.DataFrame(rows)
    sort_col = "random_R2" if df["random_R2"].notna().any() else "LOWO_R2"
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    df.to_csv(out / "final_comparison_table.csv", index=False, encoding="utf-8-sig")
    return df


def write_comparison(out: Path):
    lines: list[str] = []

    def emit(s=""):
        lines.append(s); print(s, flush=True)

    def _read(name):
        p = out / name
        return pd.read_csv(p) if p.exists() else None

    am = _read("all_model_fold_metrics.csv")
    _r = _read("random_holdout_model_summary.csv"); rsum = _r.set_index("model") if _r is not None else None
    _p = _read("lowo_pooled_summary.csv"); pooled = _p.set_index("model") if _p is not None else None
    _l = _read("lowo_macro_summary.csv"); lmac = _l.set_index("model") if _l is not None else None
    _a = _read("random_holdout_ablation_summary.csv"); abl = _a.set_index("variant") if _a is not None else None

    def rnd(m):
        return rsum.loc[m, "mean_r2_mean"] if (rsum is not None and m in rsum.index) else np.nan

    def low(m):
        return pooled.loc[m, "mean_r2"] if (pooled is not None and m in pooled.index) else np.nan

    table = build_final_table(out)

    emit("=" * 92)
    emit("FINAL COMPARISON TABLE — 55-well / 1967-stage dataset (mean target R2 / RMSE)")
    emit("=" * 92)
    emit(f"{'#':>2} {'model':26} {'category':34} {'rnd_R2':>7} {'LOWO_R2':>8} {'rnd_RMSE':>9} {'LOWO_RMSE':>9}")
    for _, r in table.iterrows():
        emit(f"{int(r['rank']):>2} {r['model']:26} {str(r['category']):34} "
             f"{fmt(r['random_R2']):>7} {fmt(r['LOWO_R2']):>8} "
             f"{fmt(r['random_RMSE'],2):>9} {fmt(r['LOWO_RMSE'],2):>9}")
    emit("\n(Saved to final_comparison_table.csv)")

    # Fair-baseline verdict
    if not np.isnan(rnd("PC-PSMoE")):
        fair_present = [m for m in FAIR_DEEP + FAIR_TREE if not np.isnan(rnd(m))]
        if fair_present:
            best = max(fair_present, key=lambda m: rnd(m))
            gap = rnd("PC-PSMoE") - rnd(best)
            emit(f"\nVERDICT (random hold-out): best fair baseline = {best} ({fmt(rnd(best))}); "
                 f"PC-PSMoE = {fmt(rnd('PC-PSMoE'))};  PC-PSMoE - best_fair = {fmt(gap)} "
                 f"=> {'PC-PSMoE better' if gap > 0 else 'PC-PSMoE NOT better'}")

    # Paper comparison (PC-PSMoE target level)
    if am is not None and ((am.protocol == "random_holdout") & (am.model == "PC-PSMoE")).any():
        pcr = am[(am.protocol == "random_holdout") & (am.model == "PC-PSMoE")].iloc[0]
        emit("\n[paper Table 2] RANDOM PC-PSMoE target-level   NEW vs paper")
        for tgt, key in [("L", "L_m"), ("W", "W_m"), ("H", "H_m"), ("mean", "mean")]:
            pr, prm = PAPER_RANDOM_TARGET[key]
            r2 = pcr["mean_r2"] if tgt == "mean" else pcr[f"{tgt}_r2"]
            rm = pcr["mean_rmse"] if tgt == "mean" else pcr[f"{tgt}_rmse"]
            emit(f"   {tgt:5} R2 {fmt(r2):>7} (paper {fmt(pr)})   RMSE {fmt(rm,2):>8} (paper {fmt(prm,2)})")

    if lmac is not None and "PC-PSMoE" in lmac.index:
        row = lmac.loc["PC-PSMoE"]
        emit("\n[paper Table 4] LOWO PC-PSMoE macro mean+/-std  NEW vs paper")
        for tgt, key in [("L", "L_m"), ("W", "W_m"), ("H", "H_m"), ("mean", "mean")]:
            pm, ps, pp = PAPER_LOWO_MACRO[key]
            mc = "mean" if tgt == "mean" else tgt
            emit(f"   {tgt:5} {fmt(row[mc+'_r2_mean'])}+/-{fmt(row[mc+'_r2_std'])} "
                 f"(paper {fmt(pm)}+/-{fmt(ps)})")

    if abl is not None:
        emit("\n[paper Table 7] ABLATION random hold-out   NEW R2 (dR2) vs paper R2 (dR2)")
        for v in ["full", "no_pkn", "no_group_dro", "no_physics", "single_expert",
                  "no_static", "no_curve"]:
            if v not in abl.index:
                continue
            nr = abl.loc[v, "mean_r2"]
            nd = abl.loc[v, "mean_r2_drop_vs_full"] if "mean_r2_drop_vs_full" in abl.columns else np.nan
            pr2, pd_ = PAPER_ABLATION[v]
            ndv = -abs(nd) if (isinstance(nd, float) and not math.isnan(nd) and v != "full") else nd
            emit(f"   {v:14} {fmt(nr):>7} ({fmt(ndv):>7}) | paper {fmt(pr2):>6} ({fmt(pd_):>7})")

    emit("\n[paired tests] PC-PSMoE vs each baseline (positive gain = PC-PSMoE better)")
    for label, fname in [("random", "random_holdout_pc_psmoe_paired_tests.csv"),
                         ("LOWO_pooled", "lowo_pc_psmoe_paired_tests.csv")]:
        t = _read(fname)
        if t is None:
            continue
        cols = [c for c in ["competitor", "mean_scaled_abs_error_gain", "pc_better_fraction",
                            "wilcoxon_p_less_error", "paired_t_p_less_error"] if c in t.columns]
        emit(f"  -- {label} --")
        emit(t[cols].to_string(index=False))

    (out / "comparison_to_paper.txt").write_text("\n".join(lines), encoding="utf-8")
    emit(f"\nSaved full report to: {out / 'comparison_to_paper.txt'}")
    return table


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="One-file full model comparison.")
    ap.add_argument("--epochs", type=int, default=100, help="Max epochs for PC-PSMoE folds.")
    ap.add_argument("--fair-epochs", type=int, default=100, help="Max epochs for fair deep baselines.")
    ap.add_argument("--seed", type=int, default=20260610)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--protocols", nargs="+", choices=["random", "lowo"],
                    default=["random", "lowo"])
    ap.add_argument("--no-fair", action="store_true", help="Disable fair baselines.")
    ap.add_argument("--fair-models", nargs="+", default=None, help="Subset of fair models.")
    ap.add_argument("--skip-ablations", action="store_true")
    ap.add_argument("--max-lowo-folds", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="8-epoch smoke test, 3 LOWO folds.")
    args = ap.parse_args()

    if args.quick:
        args.epochs = args.fair_epochs = 8
        if args.max_lowo_folds == 0:
            args.max_lowo_folds = 3

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else (args.device if args.device != "auto" else "cpu"))
    use_fair = not args.no_fair

    prevent_sleep()
    project_root = find_project_root()
    data_dir = ensure_data(project_root)
    out = args.output_dir or (project_root / "03_validation_results" / "all_comparisons_18wells")

    # ------------------------------------------------------------------
    # CACHE GUARD: fold caches are only valid for the exact dataset files
    # AND the exact training budget/seed they were produced with.  If any
    # of those changed, quarantine the whole output dir to *_stale_<ts>
    # instead of silently reusing wrong results ("(reused)" bug).
    # ------------------------------------------------------------------
    import hashlib
    fp = hashlib.md5()
    for fn in ("completion_geomechanics_parameters.xlsx",
               "treatment_curve_matrix_120step.csv"):
        p = data_dir / fn
        st = p.stat()
        fp.update(f"{fn}:{st.st_size}:{int(st.st_mtime)}".encode())
    fp.update(f"epochs={args.epochs}:fair={args.fair_epochs}:seed={args.seed}".encode())
    fingerprint = fp.hexdigest()
    fp_file = out / "cache_fingerprint.txt"
    if out.exists() and any(out.iterdir()):
        old = fp_file.read_text(encoding="utf-8").strip() if fp_file.exists() else "<missing>"
        if old != fingerprint:
            stale = out.parent / f"{out.name}_stale_{time.strftime('%Y%m%d_%H%M%S')}"
            shutil.move(str(out), str(stale))
            print(f"[cache-guard] dataset/params changed -> old results moved to {stale}")
    out.mkdir(parents=True, exist_ok=True)
    fp_file.write_text(fingerprint, encoding="utf-8")
    log_path = out / "run_log.txt"

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    pc = load_model_module()
    pc.seed_everything(args.seed)
    raw = pc.load_raw_data(data_dir, pkn_source="raw")
    log(f"data_dir={data_dir}")
    log(f"samples={len(raw.y)} wells={len(np.unique(raw.wells))} "
        f"epochs={args.epochs} fair_epochs={args.fair_epochs} fair={use_fair} device={device}")

    all_rows: list[dict[str, Any]] = []

    # ---------- RANDOM ----------
    if "random" in args.protocols:
        log("=== A. RANDOM HOLD-OUT ===")
        rsplit = pc.collect_splits("random", raw, repeats=1, group_folds=5, seed=args.seed)[0]
        variants = ["full"] if args.skip_ablations else ABLATION_VARIANTS
        ablation_rows, full_mean = [], None
        for v in variants:
            t0 = time.time()
            m = run_pc_fold(pc, raw, rsplit, v, out / "pc" / "random" / v,
                            1, args.epochs, args.seed, device)
            tag = "reused" if m.get("_reused") else f"{time.time()-t0:.0f}s"
            log(f"  [random/{v}] mean_r2={m['mean_r2']:.4f} ({tag})")
            ablation_rows.append({"variant": v, "L_m_r2": m["L_m_r2"], "W_m_r2": m["W_m_r2"],
                                  "H_m_r2": m["H_m_r2"], "mean_r2": m["mean_r2"],
                                  "mean_rmse": m["mean_rmse"], "mean_mae": m["mean_mae"]})
            if v == "full":
                full_mean = m["mean_r2"]
        ablation_df = pd.DataFrame(ablation_rows)
        if full_mean is not None:
            ablation_df["mean_r2_drop_vs_full"] = full_mean - ablation_df["mean_r2"]
        ablation_df.to_csv(out / "random_holdout_ablation_summary.csv", index=False, encoding="utf-8-sig")

        y_rnd, pc_rnd = load_pc_predictions(out / "pc" / "random" / "full" / rsplit.name)
        rnd_fit = np.sort(np.r_[rsplit.train_idx, rsplit.val_idx])
        all_rows.append(metric_row("random_holdout", rsplit.name, "PC-PSMoE", y_rnd, pc_rnd))
        log("  standard baselines ...")
        comp = strong_baseline_predictions(pc, raw, rnd_fit, rsplit.test_idx, seed=20260611)
        if use_fair:
            log("  fair baselines ...")
            comp.update(fair_baseline_predictions(
                pc, raw, rsplit.train_idx, rsplit.val_idx, rsplit.test_idx,
                cache_dir=out / "fair_cache" / "random", fold_name=rsplit.name,
                epochs=args.fair_epochs, seed=args.seed, device=device,
                models=args.fair_models, log=log))
        for model, pred in comp.items():
            all_rows.append(metric_row("random_holdout", rsplit.name, model, y_rnd, pred))
        bootstrap_ci(y_rnd, pc_rnd).to_csv(out / "random_holdout_pc_psmoe_bootstrap_ci.csv",
                                           index=False, encoding="utf-8-sig")
        paired_tests("random_holdout", "PC-PSMoE", y_rnd, pc_rnd, comp).to_csv(
            out / "random_holdout_pc_psmoe_paired_tests.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(all_rows).to_csv(out / "all_model_fold_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- LOWO ----------
    store, lowo_splits = {}, []
    if "lowo" in args.protocols:
        log("=== B. LEAVE-ONE-WELL-OUT ===")
        lowo_splits = pc.collect_splits("lowo", raw, repeats=1, group_folds=5, seed=args.seed)
        if args.max_lowo_folds > 0:
            lowo_splits = lowo_splits[: args.max_lowo_folds]
        log(f"  LOWO folds: {len(lowo_splits)}")
        # Memory-bounded operation: the long-lived process grows its commit
        # charge fold over fold (Windows then silently expands pagefile.sys
        # until C: hits zero and the box freezes).  After FOLDS_PER_PROCESS
        # freshly-trained folds we exit with code 75; auto_resume_lowo.py
        # treats 75 as a planned recycle and restarts immediately — caches
        # make the replay cheap.
        import gc
        recycle_after = int(os.environ.get("FOLDS_PER_PROCESS", "0"))
        trained_folds = 0
        for fn, split in enumerate(lowo_splits, start=1):
            t0 = time.time()
            m = run_pc_fold(pc, raw, split, "full", out / "pc" / "lowo" / "full",
                            fn, args.epochs, args.seed, device)
            y_true, pc_pred = load_pc_predictions(out / "pc" / "lowo" / "full" / split.name)
            all_rows.append(metric_row("LOWO", split.name, "PC-PSMoE", y_true, pc_pred))
            store.setdefault("PC-PSMoE", {"y": [], "pred": []})
            store["PC-PSMoE"]["y"].append(y_true); store["PC-PSMoE"]["pred"].append(pc_pred)
            fit = np.sort(np.r_[split.train_idx, split.val_idx])
            # strong (tree/linear) baselines were recomputed on every replay
            # (~80s/fold), making restarts expensive — cache them like the
            # fair baselines.  The _done flag guards against partial writes.
            scache = out / "strong_cache" / "lowo" / split.name
            sdone = scache / "_done.flag"
            if sdone.exists():
                comp = {p.stem: np.load(p) for p in sorted(scache.glob("*.npy"))}
            else:
                comp = strong_baseline_predictions(pc, raw, fit, split.test_idx, seed=20260611 + len(all_rows))
                scache.mkdir(parents=True, exist_ok=True)
                for model, pred in comp.items():
                    np.save(scache / f"{model}.npy", pred)
                sdone.write_text("ok", encoding="utf-8")
            if use_fair:
                comp.update(fair_baseline_predictions(
                    pc, raw, split.train_idx, split.val_idx, split.test_idx,
                    cache_dir=out / "fair_cache" / "lowo", fold_name=split.name,
                    epochs=args.fair_epochs, seed=args.seed + fn * 101, device=device,
                    models=args.fair_models, log=log))
            for model, pred in comp.items():
                all_rows.append(metric_row("LOWO", split.name, model, y_true, pred))
                store.setdefault(model, {"y": [], "pred": []})
                store[model]["y"].append(y_true); store[model]["pred"].append(pred)
            tag = "reused" if m.get("_reused") else f"{time.time()-t0:.0f}s"
            log(f"  [{fn:02d}/{len(lowo_splits)}] {split.name} PC mean_r2={m['mean_r2']:.4f} ({tag})")
            pd.DataFrame(all_rows).to_csv(out / "all_model_fold_metrics.csv", index=False, encoding="utf-8-sig")
            if not m.get("_reused"):
                trained_folds += 1
                gc.collect()
                if 0 < recycle_after <= trained_folds and fn < len(lowo_splits):
                    log(f"  [recycle] {trained_folds} folds trained in this process; "
                        f"exiting 75 for a fresh process (auto_resume restarts immediately).")
                    sys.exit(75)

    # ---------- Summaries ----------
    all_metrics = pd.DataFrame(all_rows)
    all_metrics.to_csv(out / "all_model_fold_metrics.csv", index=False, encoding="utf-8-sig")
    if "random" in args.protocols:
        summarize_model_rows(all_metrics, "random_holdout").to_csv(
            out / "random_holdout_model_summary.csv", index=False, encoding="utf-8-sig")
    if "lowo" in args.protocols and store:
        summarize_model_rows(all_metrics, "LOWO").to_csv(
            out / "lowo_macro_summary.csv", index=False, encoding="utf-8-sig")
        pooled_rows, competitors, y_pc, pc_pool = [], {}, None, None
        for model, s in store.items():
            yp, pp = np.vstack(s["y"]), np.vstack(s["pred"])
            pooled_rows.append(metric_row("LOWO_pooled", "pooled", model, yp, pp))
            if model == "PC-PSMoE":
                y_pc, pc_pool = yp, pp
            else:
                competitors[model] = pp
        pd.DataFrame(pooled_rows).sort_values("mean_r2", ascending=False).to_csv(
            out / "lowo_pooled_summary.csv", index=False, encoding="utf-8-sig")
        if y_pc is not None:
            bootstrap_ci(y_pc, pc_pool).to_csv(out / "lowo_pc_psmoe_bootstrap_ci.csv",
                                               index=False, encoding="utf-8-sig")
            paired_tests("LOWO_pooled", "PC-PSMoE", y_pc, pc_pool, competitors).to_csv(
                out / "lowo_pc_psmoe_paired_tests.csv", index=False, encoding="utf-8-sig")

    # ---------- Final table + report + Excel ----------
    log("=== writing final comparison table ===")
    table = write_comparison(out)

    sheet_files = [("final_table", "final_comparison_table.csv"),
                   ("ablation_random", "random_holdout_ablation_summary.csv"),
                   ("random_baselines", "random_holdout_model_summary.csv"),
                   ("lowo_macro", "lowo_macro_summary.csv"),
                   ("lowo_pooled", "lowo_pooled_summary.csv"),
                   ("all_fold_metrics", "all_model_fold_metrics.csv"),
                   ("random_bootstrap", "random_holdout_pc_psmoe_bootstrap_ci.csv"),
                   ("random_paired", "random_holdout_pc_psmoe_paired_tests.csv"),
                   ("lowo_bootstrap", "lowo_pc_psmoe_bootstrap_ci.csv"),
                   ("lowo_paired", "lowo_pc_psmoe_paired_tests.csv")]
    with pd.ExcelWriter(out / "all_comparisons_18wells.xlsx", engine="openpyxl") as w:
        for sheet, fname in sheet_files:
            if (out / fname).exists():
                pd.read_csv(out / fname).to_excel(w, sheet_name=sheet, index=False)

    (out / "run_manifest.json").write_text(json.dumps({
        "dataset": "55 wells / 1967 stages", "n_samples": int(len(raw.y)),
        "n_wells": int(len(np.unique(raw.wells))), "epochs": args.epochs,
        "fair_epochs": args.fair_epochs, "fair_baselines": use_fair,
        "protocols": args.protocols, "n_lowo_folds": len(lowo_splits),
        "seed": args.seed, "device": str(device),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Saved all outputs to: {out}")


if __name__ == "__main__":
    main()
