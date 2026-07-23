from __future__ import annotations

import argparse
import copy
import json
import math
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset


MODEL_NAME = "Physics-constrained Probabilistic Soft-domain Mixture-of-Experts"
MODEL_ABBR = "PC-PSMoE"
TARGETS = ["L_m", "W_m", "H_m"]
PKN_CALIBRATED_COLUMNS = [
    "PKN_cal_L_m",
    "PKN_cal_W_m",
    "PKN_cal_H_m",
]
PKN_RAW_COLUMNS = [
    "PKN_L_m",
    "PKN_W_m",
    "PKN_H_m",
]
SEQUENCE_COLUMNS = [
    "time_norm",
    "rate_m3_min",
    "pressure_mpa",
    "sand_concentration",
    "cumulative_rate_norm",
    "pressure_gradient",
    "pressure_rate_power",
    "sand_rate",
]

# Directly recorded geology/engineering variables only. Well identity, stage
# order, curve summaries, target-equivalent columns, PKN outputs, and derived
# fracture geometry variables are deliberately excluded.
STATIC_NUMERIC_COLUMNS = [
    "垂深m",
    "簇数",
    "孔数",
    "相位角",
    "段间距",
    "主压裂滑溜水用量",
    "主压裂冻胶",
    "主压裂总液量",
    "入井总液量",
    "实际总液量",
    "70-140目粉砂m³",
    "40-70目石英砂m³",
    "单段支撑剂总量m³",
    "测试停泵压力MPa",
    "近井摩阻MPa",
    "主压裂排量\nm3/min",
    "主压裂排量min\nm3/min",
    "主压裂施工压力\nMPa",
    "滑溜水比例%",
    "砂液比\n%",
    "压后停泵压力MPa",
    "CO2\nt",
    "杨氏模量",
    "泊松比",
]
STATIC_CATEGORICAL_COLUMNS = ["构造位置", "层位", "枪型/弹型", "孔密"]

PHYSICS_PROXY_COLUMNS = [
    "实际总液量",
    "单段支撑剂总量m³",
    "主压裂排量\nm3/min",
    "主压裂施工压力\nMPa",
]


@dataclass
class ModelConfig:
    static_hidden: int = 96
    sequence_width: int = 96
    pkn_hidden: int = 48
    fusion_width: int = 192
    expert_width: int = 128
    n_domains: int = 6
    transformer_layers: int = 3
    transformer_heads: int = 8
    dropout: float = 0.12
    gate_temperature: float = 0.85
    min_log_sigma: float = -3.8
    max_log_sigma: float = 0.65


@dataclass
class TrainConfig:
    seed: int = 20260610
    epochs: int = 240
    batch_size: int = 128
    learning_rate: float = 7e-4
    weight_decay: float = 2e-4
    patience: int = 40
    grad_clip: float = 3.0
    num_workers: int = 0
    lambda_residual: float = 0.25
    lambda_bounds: float = 0.08
    lambda_volume: float = 0.035
    lambda_monotonic: float = 0.018
    lambda_gate_balance: float = 0.018
    lambda_gate_entropy: float = 0.008
    lambda_expert_diversity: float = 0.008
    lambda_group_dro: float = 0.18
    group_dro_temperature: float = 0.35
    checkpoint_ensemble_k: int = 1  # 1 = best single checkpoint by validation NLL
    model_ensemble_k: int = 1       # 1 = single model
    target_gate_entropy_ratio: float = 0.55


@dataclass
class AblationConfig:
    variant: str = "full"
    use_static: bool = True
    use_curve: bool = True
    use_pkn: bool = True
    use_physics_losses: bool = True


@dataclass
class RawData:
    sample_ids: np.ndarray
    wells: np.ndarray
    stages: np.ndarray
    operation_order: np.ndarray
    static_frame: pd.DataFrame
    pkn_raw: np.ndarray
    sequence_raw: np.ndarray
    y: np.ndarray


@dataclass
class PreparedData:
    static: np.ndarray
    sequence: np.ndarray
    pkn: np.ndarray
    y: np.ndarray
    volume_reference: np.ndarray
    dose_proxy: np.ndarray
    group_ids: np.ndarray


@dataclass
class FoldSplit:
    name: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def default_data_dir() -> Path:
    delivery_root = Path(__file__).resolve().parents[2]
    candidates = [
        delivery_root / "data" / "field_records",
        delivery_root / "01_final_data" / "field_records",
    ]
    for candidate in candidates:
        if (candidate / "completion_geomechanics_parameters.xlsx").exists():
            return candidate
    return candidates[0]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def _align(frame: pd.DataFrame, sample_ids: np.ndarray, name: str) -> pd.DataFrame:
    if "sample_id" not in frame:
        raise ValueError(f"{name} has no sample_id column.")
    aligned = pd.DataFrame({"sample_id": sample_ids}).merge(
        frame, on="sample_id", how="left", validate="one_to_one"
    )
    if aligned.isna().all(axis=1).any():
        raise ValueError(f"{name} could not be aligned to the sample index.")
    return aligned


def load_raw_data(data_dir: Path, pkn_source: str = "raw") -> RawData:
    workbook = data_dir / "completion_geomechanics_parameters.xlsx"
    sequence_csv = data_dir / "treatment_curve_matrix_120step.csv"
    if not workbook.exists() or not sequence_csv.exists():
        raise FileNotFoundError(
            f"Expected workbook and sequence matrix under {data_dir}."
        )

    index_df = pd.read_excel(workbook, sheet_name=0)
    static_df = pd.read_excel(workbook, sheet_name=1)
    pkn_df = pd.read_excel(workbook, sheet_name=2)
    sample_ids = index_df["sample_id"].astype(str).to_numpy()
    static_df = _align(static_df, sample_ids, "static sheet")
    pkn_df = _align(pkn_df, sample_ids, "PKN sheet")

    requested_static = STATIC_NUMERIC_COLUMNS + STATIC_CATEGORICAL_COLUMNS
    missing_static = [c for c in requested_static if c not in static_df.columns]
    if missing_static:
        warnings.warn(f"Missing static columns will be skipped: {missing_static}")
    static_columns = [c for c in requested_static if c in static_df.columns]
    static_frame = static_df[static_columns].copy()
    for column in STATIC_NUMERIC_COLUMNS:
        if column in static_frame:
            static_frame[column] = pd.to_numeric(
                static_frame[column], errors="coerce"
            )
    for column in STATIC_CATEGORICAL_COLUMNS:
        if column in static_frame:
            static_frame[column] = (
                static_frame[column]
                .astype("string")
                .fillna("__MISSING__")
                .astype(object)
            )

    pkn_columns = PKN_RAW_COLUMNS if pkn_source == "raw" else PKN_CALIBRATED_COLUMNS
    missing_pkn = [c for c in pkn_columns if c not in pkn_df.columns]
    if missing_pkn:
        raise ValueError(f"Missing PKN columns: {missing_pkn}")
    pkn_raw = (
        pkn_df[pkn_columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype=np.float64)
    )
    y = pkn_df[TARGETS].to_numpy(dtype=np.float64)

    usecols = ["sample_id", "time_step"] + SEQUENCE_COLUMNS
    sequence_df = pd.read_csv(sequence_csv, usecols=usecols)
    sequence_df["sample_id"] = pd.Categorical(
        sequence_df["sample_id"].astype(str), categories=sample_ids, ordered=True
    )
    sequence_df = sequence_df.sort_values(["sample_id", "time_step"])
    counts = sequence_df.groupby("sample_id", observed=False).size().to_numpy()
    if len(counts) != len(sample_ids) or len(np.unique(counts)) != 1:
        raise ValueError("Every sample must have the same number of sequence steps.")
    n_steps = int(counts[0])
    sequence_raw = sequence_df[SEQUENCE_COLUMNS].to_numpy(dtype=np.float64)
    sequence_raw = sequence_raw.reshape(len(sample_ids), n_steps, len(SEQUENCE_COLUMNS))

    return RawData(
        sample_ids=sample_ids,
        wells=index_df["well"].astype(str).to_numpy(),
        stages=pd.to_numeric(index_df["stage"], errors="coerce").fillna(-1).to_numpy(),
        operation_order=pd.to_numeric(
            index_df["operation_order"]
            if "operation_order" in index_df
            else index_df["stage"],
            errors="coerce",
        )
        .fillna(-1)
        .to_numpy(),
        static_frame=static_frame,
        pkn_raw=pkn_raw,
        sequence_raw=sequence_raw,
        y=y,
    )


class FoldPreprocessor:
    """All fitted state is learned from the training fold only."""

    def __init__(self, ablation: AblationConfig | None = None) -> None:
        self.ablation = ablation or AblationConfig()
        self.static_transformer: ColumnTransformer | None = None
        self.sequence_mean: np.ndarray | None = None
        self.sequence_std: np.ndarray | None = None
        self.pkn_imputer: SimpleImputer | None = None
        self.pkn_calibrators: list[Ridge] = []
        self.pkn_log_mean: np.ndarray | None = None
        self.pkn_log_std: np.ndarray | None = None
        self.volume_model: Ridge | None = None
        self.proxy_imputer: SimpleImputer | None = None
        self.proxy_scaler: StandardScaler | None = None
        self.bounds: tuple[np.ndarray, np.ndarray] | None = None
        self.group_map: dict[str, int] = {}
        self.static_feature_names: list[str] = []

    @staticmethod
    def _physics_proxy(frame: pd.DataFrame) -> np.ndarray:
        values = []
        for col in PHYSICS_PROXY_COLUMNS:
            if col in frame:
                values.append(pd.to_numeric(frame[col], errors="coerce").to_numpy())
            else:
                values.append(np.full(len(frame), np.nan))
        x = np.column_stack(values).astype(np.float64)
        x[:, :2] = np.log1p(np.maximum(x[:, :2], 0.0))
        x[:, 2:] = np.log1p(np.maximum(x[:, 2:], 0.0))
        return x

    def fit_transform(self, raw: RawData, train_idx: np.ndarray) -> PreparedData:
        if self.ablation.use_static:
            numeric = [c for c in STATIC_NUMERIC_COLUMNS if c in raw.static_frame]
            categorical = [c for c in STATIC_CATEGORICAL_COLUMNS if c in raw.static_frame]
            numeric_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scaler", StandardScaler()),
                ]
            )
            categorical_pipe = Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent",
                            keep_empty_features=True,
                        ),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                            min_frequency=2,
                        ),
                    ),
                ]
            )
            self.static_transformer = ColumnTransformer(
                [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)],
                remainder="drop",
                verbose_feature_names_out=False,
            )
            self.static_transformer.fit(raw.static_frame.iloc[train_idx])
            static = self.static_transformer.transform(raw.static_frame).astype(np.float32)
            self.static_feature_names = list(self.static_transformer.get_feature_names_out())
        else:
            self.static_transformer = None
            static = np.zeros((len(raw.y), 1), dtype=np.float32)
            self.static_feature_names = ["static_branch_disabled"]

        if self.ablation.use_curve:
            train_sequence = raw.sequence_raw[train_idx]
            self.sequence_mean = np.nanmean(train_sequence, axis=(0, 1), keepdims=True)
            self.sequence_std = np.nanstd(train_sequence, axis=(0, 1), keepdims=True)
            self.sequence_std = np.maximum(self.sequence_std, 1e-6)
            sequence = np.nan_to_num(
                (raw.sequence_raw - self.sequence_mean) / self.sequence_std,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32)
        else:
            self.sequence_mean = np.zeros((1, 1, raw.sequence_raw.shape[2]), dtype=np.float64)
            self.sequence_std = np.ones((1, 1, raw.sequence_raw.shape[2]), dtype=np.float64)
            sequence = np.zeros_like(raw.sequence_raw, dtype=np.float32)

        train_lo = np.quantile(raw.y[train_idx], 0.005, axis=0)
        train_hi = np.quantile(raw.y[train_idx], 0.995, axis=0)
        span = np.maximum(train_hi - train_lo, 1e-4)
        lo = np.maximum(train_lo - 0.10 * span, 1e-4)
        hi = train_hi + 0.10 * span
        self.bounds = (lo, hi)

        if self.ablation.use_pkn:
            self.pkn_imputer = SimpleImputer(strategy="median")
            self.pkn_imputer.fit(raw.pkn_raw[train_idx])
            pkn_raw = np.maximum(self.pkn_imputer.transform(raw.pkn_raw), 1e-5)
            log_pkn_raw = np.log(pkn_raw)
            log_y = np.log(np.maximum(raw.y, 1e-5))
            calibrated_log = np.zeros_like(log_y)
            self.pkn_calibrators = []
            for target_i in range(3):
                model = Ridge(alpha=2.0)
                model.fit(log_pkn_raw[train_idx], log_y[train_idx, target_i])
                calibrated_log[:, target_i] = model.predict(log_pkn_raw)
                self.pkn_calibrators.append(model)
            pkn = np.clip(np.exp(calibrated_log), lo, hi)
        else:
            self.pkn_imputer = None
            self.pkn_calibrators = []
            train_median = np.median(raw.y[train_idx], axis=0)
            pkn = np.tile(np.clip(train_median, lo, hi), (len(raw.y), 1))

        pkn_log = np.log(pkn)
        self.pkn_log_mean = pkn_log[train_idx].mean(axis=0, keepdims=True)
        self.pkn_log_std = np.maximum(
            pkn_log[train_idx].std(axis=0, keepdims=True), 1e-6
        )

        proxy_raw = self._physics_proxy(raw.static_frame)
        self.proxy_imputer = SimpleImputer(strategy="median")
        self.proxy_scaler = StandardScaler()
        self.proxy_imputer.fit(proxy_raw[train_idx])
        proxy = self.proxy_imputer.transform(proxy_raw)
        self.proxy_scaler.fit(proxy[train_idx])
        proxy_scaled = self.proxy_scaler.transform(proxy)
        dose_proxy = proxy_scaled.mean(axis=1)

        log_volume = np.log(
            np.maximum(raw.y[:, 0] * (raw.y[:, 1] / 1000.0) * raw.y[:, 2], 1e-8)
        )
        self.volume_model = Ridge(alpha=4.0)
        self.volume_model.fit(proxy_scaled[train_idx], log_volume[train_idx])
        volume_reference = self.volume_model.predict(proxy_scaled)

        train_wells = sorted(np.unique(raw.wells[train_idx]))
        self.group_map = {well: i for i, well in enumerate(train_wells)}
        unknown_id = len(self.group_map)
        group_ids = np.array(
            [self.group_map.get(well, unknown_id) for well in raw.wells], dtype=np.int64
        )

        return PreparedData(
            static=static,
            sequence=sequence,
            pkn=pkn.astype(np.float32),
            y=raw.y.astype(np.float32),
            volume_reference=volume_reference.astype(np.float32),
            dose_proxy=dose_proxy.astype(np.float32),
            group_ids=group_ids,
        )

    def manifest(self) -> dict[str, Any]:
        if self.bounds is None:
            raise RuntimeError("Preprocessor has not been fitted.")
        return {
            "static_features": self.static_feature_names,
            "sequence_channels": SEQUENCE_COLUMNS,
            "pkn_calibration": [
                {
                    "coef": model.coef_.tolist(),
                    "intercept": float(model.intercept_),
                }
                for model in self.pkn_calibrators
            ],
            "engineering_bounds": {
                target: [float(self.bounds[0][i]), float(self.bounds[1][i])]
                for i, target in enumerate(TARGETS)
            },
            "training_well_groups": self.group_map,
        }


class FractureDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, data: PreparedData) -> None:
        self.static = torch.from_numpy(data.static)
        self.sequence = torch.from_numpy(data.sequence)
        self.pkn = torch.from_numpy(data.pkn)
        self.y = torch.from_numpy(data.y)
        self.volume_reference = torch.from_numpy(data.volume_reference)
        self.dose_proxy = torch.from_numpy(data.dose_proxy)
        self.group_ids = torch.from_numpy(data.group_ids)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "static": self.static[index],
            "sequence": self.sequence[index],
            "pkn": self.pkn[index],
            "y": self.y[index],
            "volume_reference": self.volume_reference[index],
            "dose_proxy": self.dose_proxy[index],
            "group_id": self.group_ids[index],
            "row_index": torch.tensor(index, dtype=torch.long),
        }


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class TemporalCurveEncoder(nn.Module):
    """Encodes the complete second-scale sequence without summary compression."""

    def __init__(
        self,
        n_channels: int,
        width: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        max_steps: int = 512,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(n_channels, width)
        self.local_conv_3 = nn.Conv1d(width, width, kernel_size=3, padding=1)
        self.local_conv_7 = nn.Conv1d(width, width, kernel_size=7, padding=3)
        self.local_norm = nn.LayerNorm(width)
        self.position = nn.Parameter(torch.zeros(1, max_steps, width))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=n_heads,
            dim_feedforward=width * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.attention_score = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width // 2),
            nn.Tanh(),
            nn.Linear(width // 2, 1),
        )
        self.output_norm = nn.LayerNorm(width)

    def forward(self, sequence: Tensor) -> tuple[Tensor, Tensor]:
        n_steps = sequence.shape[1]
        if n_steps > self.position.shape[1]:
            raise ValueError(f"Sequence length {n_steps} exceeds configured maximum.")
        x = self.input_projection(sequence)
        local = x.transpose(1, 2)
        local = F.gelu(self.local_conv_3(local)) + F.gelu(self.local_conv_7(local))
        x = self.local_norm(x + local.transpose(1, 2))
        x = self.transformer(x + self.position[:, :n_steps])
        attention = torch.softmax(self.attention_score(x).squeeze(-1), dim=1)
        pooled = torch.sum(x * attention.unsqueeze(-1), dim=1)
        return self.output_norm(pooled), attention


class ProbabilisticExpert(nn.Module):
    def __init__(self, input_width: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(hidden, dropout),
            ResidualBlock(hidden, dropout),
            nn.LayerNorm(hidden),
        )
        self.mean_head = nn.Linear(hidden, 3)
        self.scale_head = nn.Linear(hidden, 3)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.network(x)
        return self.mean_head(h), self.scale_head(h)


class PCPSMoE(nn.Module):
    def __init__(
        self,
        static_dim: int,
        sequence_channels: int,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, config.static_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            ResidualBlock(config.static_hidden, config.dropout),
            nn.LayerNorm(config.static_hidden),
        )
        self.sequence_encoder = TemporalCurveEncoder(
            n_channels=sequence_channels,
            width=config.sequence_width,
            n_layers=config.transformer_layers,
            n_heads=config.transformer_heads,
            dropout=config.dropout,
        )
        self.pkn_encoder = nn.Sequential(
            nn.Linear(3, config.pkn_hidden),
            nn.GELU(),
            ResidualBlock(config.pkn_hidden, config.dropout),
            nn.LayerNorm(config.pkn_hidden),
        )
        combined_width = (
            config.static_hidden + config.sequence_width + config.pkn_hidden
        )
        self.fusion = nn.Sequential(
            nn.Linear(combined_width, config.fusion_width),
            nn.GELU(),
            nn.Dropout(config.dropout),
            ResidualBlock(config.fusion_width, config.dropout),
            nn.LayerNorm(config.fusion_width),
        )
        self.gate = nn.Sequential(
            nn.Linear(config.fusion_width, config.fusion_width // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_width // 2, config.n_domains),
        )
        self.experts = nn.ModuleList(
            [
                ProbabilisticExpert(
                    config.fusion_width,
                    config.expert_width,
                    config.dropout,
                )
                for _ in range(config.n_domains)
            ]
        )

    def forward(self, static: Tensor, sequence: Tensor, pkn: Tensor) -> dict[str, Tensor]:
        static_embedding = self.static_encoder(static)
        sequence_embedding, sequence_attention = self.sequence_encoder(sequence)
        pkn_embedding = self.pkn_encoder(torch.log(torch.clamp(pkn, min=1e-5)))
        fused = self.fusion(
            torch.cat([static_embedding, sequence_embedding, pkn_embedding], dim=1)
        )
        gate_logits = self.gate(fused) / self.config.gate_temperature
        gate_prob = torch.softmax(gate_logits, dim=1)

        means, raw_scales = zip(*(expert(fused) for expert in self.experts))
        residual_mean = torch.stack(means, dim=1)
        raw_scale = torch.stack(raw_scales, dim=1)
        log_sigma = torch.clamp(
            raw_scale,
            min=self.config.min_log_sigma,
            max=self.config.max_log_sigma,
        )
        sigma = torch.exp(log_sigma)

        mixed_residual = torch.sum(gate_prob.unsqueeze(-1) * residual_mean, dim=1)
        log_pkn = torch.log(torch.clamp(pkn, min=1e-5))
        log_prediction = log_pkn + mixed_residual
        prediction = torch.exp(log_prediction)

        component_log_y = log_pkn.unsqueeze(1) + residual_mean
        component_second = torch.exp(
            2.0 * component_log_y + sigma.square()
        )
        component_mean = torch.exp(component_log_y + 0.5 * sigma.square())
        mixture_mean = torch.sum(gate_prob.unsqueeze(-1) * component_mean, dim=1)
        mixture_second = torch.sum(
            gate_prob.unsqueeze(-1) * component_second, dim=1
        )
        mixture_var = torch.clamp(
            mixture_second - mixture_mean.square(), min=1e-8
        )
        relative_log_std = torch.sqrt(
            torch.log1p(mixture_var / torch.clamp(mixture_mean.square(), min=1e-8))
        )

        return {
            "prediction": prediction,
            "mixture_mean": mixture_mean,
            "log_prediction": log_prediction,
            "log_std": relative_log_std,
            "gate_prob": gate_prob,
            "gate_logits": gate_logits,
            "expert_mean": residual_mean,
            "expert_sigma": sigma,
            "sequence_attention": sequence_attention,
        }


def mixture_nll(
    outputs: dict[str, Tensor],
    y: Tensor,
    pkn: Tensor,
) -> Tensor:
    log_y = torch.log(torch.clamp(y, min=1e-5))
    log_pkn = torch.log(torch.clamp(pkn, min=1e-5))
    target_residual = log_y - log_pkn
    mu = outputs["expert_mean"]
    sigma = outputs["expert_sigma"]
    z = (target_residual.unsqueeze(1) - mu) / sigma
    log_prob_targets = -0.5 * z.square() - torch.log(sigma) - 0.5 * math.log(
        2.0 * math.pi
    )
    log_prob_component = log_prob_targets.sum(dim=2)
    log_gate = torch.log(torch.clamp(outputs["gate_prob"], min=1e-8))
    return -torch.logsumexp(log_gate + log_prob_component, dim=1)


def grouped_robust_loss(
    sample_loss: Tensor,
    group_ids: Tensor,
    strength: float,
    temperature: float,
) -> Tensor:
    group_losses = []
    for group in torch.unique(group_ids):
        mask = group_ids == group
        if mask.any():
            group_losses.append(sample_loss[mask].mean())
    if len(group_losses) <= 1:
        return sample_loss.mean()
    stacked = torch.stack(group_losses)
    smooth_worst = temperature * torch.logsumexp(stacked / temperature, dim=0)
    return (1.0 - strength) * sample_loss.mean() + strength * smooth_worst


def physics_constrained_loss(
    outputs: dict[str, Tensor],
    batch: dict[str, Tensor],
    bounds: tuple[Tensor, Tensor],
    config: TrainConfig,
    n_domains: int,
) -> tuple[Tensor, dict[str, float]]:
    sample_nll = mixture_nll(outputs, batch["y"], batch["pkn"])
    data_loss = grouped_robust_loss(
        sample_nll,
        batch["group_id"],
        strength=config.lambda_group_dro,
        temperature=config.group_dro_temperature,
    )

    target_residual = torch.log(torch.clamp(batch["y"], min=1e-5)) - torch.log(
        torch.clamp(batch["pkn"], min=1e-5)
    )
    mixed_residual = outputs["log_prediction"] - torch.log(
        torch.clamp(batch["pkn"], min=1e-5)
    )
    residual_loss = F.smooth_l1_loss(mixed_residual, target_residual)

    lower, upper = bounds
    log_prediction = outputs["log_prediction"]
    bound_loss = (
        F.relu(torch.log(lower) - log_prediction).square()
        + F.relu(log_prediction - torch.log(upper)).square()
    ).mean()

    log_volume = (
        log_prediction[:, 0]
        + log_prediction[:, 1]
        - math.log(1000.0)
        + log_prediction[:, 2]
    )
    volume_loss = F.smooth_l1_loss(log_volume, batch["volume_reference"])

    dose = batch["dose_proxy"]
    dose_centered = dose - dose.mean()
    volume_centered = log_volume - log_volume.mean()
    corr = torch.sum(dose_centered * volume_centered) / (
        torch.sqrt(torch.sum(dose_centered.square()) + 1e-6)
        * torch.sqrt(torch.sum(volume_centered.square()) + 1e-6)
    )
    monotonic_loss = F.relu(-corr).square()

    gate_prob = outputs["gate_prob"]
    mean_gate = gate_prob.mean(dim=0)
    gate_balance = torch.sum(
        mean_gate * torch.log(torch.clamp(mean_gate * n_domains, min=1e-8))
    )
    entropy = -torch.sum(
        gate_prob * torch.log(torch.clamp(gate_prob, min=1e-8)), dim=1
    ).mean()
    entropy_target = config.target_gate_entropy_ratio * math.log(n_domains)
    gate_entropy = (entropy - entropy_target) ** 2

    expert_spread = outputs["expert_mean"].var(dim=1, unbiased=False).mean()
    expert_diversity = F.relu(0.015 - expert_spread)

    total = (
        data_loss
        + config.lambda_residual * residual_loss
        + config.lambda_bounds * bound_loss
        + config.lambda_volume * volume_loss
        + config.lambda_monotonic * monotonic_loss
        + config.lambda_gate_balance * gate_balance
        + config.lambda_gate_entropy * gate_entropy
        + config.lambda_expert_diversity * expert_diversity
    )
    diagnostics = {
        "loss": float(total.detach()),
        "nll": float(sample_nll.mean().detach()),
        "residual": float(residual_loss.detach()),
        "bounds": float(bound_loss.detach()),
        "volume": float(volume_loss.detach()),
        "monotonic": float(monotonic_loss.detach()),
        "gate_balance": float(gate_balance.detach()),
        "gate_entropy": float(entropy.detach()),
        "expert_spread": float(expert_spread.detach()),
    }
    return total, diagnostics


def move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def run_epoch(
    model: PCPSMoE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    bounds: tuple[Tensor, Tensor],
    train_config: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    diagnostics = []
    for batch_cpu in loader:
        batch = move_batch(batch_cpu, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(batch["static"], batch["sequence"], batch["pkn"])
            loss, diag = physics_constrained_loss(
                outputs,
                batch,
                bounds,
                train_config,
                model.config.n_domains,
            )
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
                optimizer.step()
        diagnostics.append(diag)
    return mean_dict(diagnostics)


@torch.no_grad()
def predict(
    model: PCPSMoE,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    store: dict[str, list[np.ndarray]] = {
        "row_index": [],
        "y": [],
        "prediction": [],
        "mixture_mean": [],
        "log_std": [],
        "gate_prob": [],
        "sequence_attention": [],
    }
    for batch_cpu in loader:
        batch = move_batch(batch_cpu, device)
        outputs = model(batch["static"], batch["sequence"], batch["pkn"])
        store["row_index"].append(batch["row_index"].cpu().numpy())
        store["y"].append(batch["y"].cpu().numpy())
        for key in [
            "prediction",
            "mixture_mean",
            "log_std",
            "gate_prob",
            "sequence_attention",
        ]:
            store[key].append(outputs[key].cpu().numpy())
    result = {key: np.concatenate(value, axis=0) for key, value in store.items()}
    order = np.argsort(result["row_index"])
    return {key: value[order] for key, value in result.items()}


def make_loader(
    dataset: FractureDataset,
    indices: np.ndarray,
    config: TrainConfig,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def fit_one_fold(
    raw: RawData,
    split: FoldSplit,
    model_config: ModelConfig,
    train_config: TrainConfig,
    ablation_config: AblationConfig,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    fold_dir = output_dir / split.name
    fold_dir.mkdir(parents=True, exist_ok=True)
    preprocessor = FoldPreprocessor(ablation_config)
    prepared = preprocessor.fit_transform(raw, split.train_idx)
    dataset = FractureDataset(prepared)
    train_loader = make_loader(dataset, split.train_idx, train_config, shuffle=True)
    val_loader = make_loader(dataset, split.val_idx, train_config, shuffle=False)
    test_loader = make_loader(dataset, split.test_idx, train_config, shuffle=False)

    if preprocessor.bounds is None:
        raise RuntimeError("Training bounds were not fitted.")
    bounds = (
        torch.tensor(preprocessor.bounds[0], dtype=torch.float32, device=device),
        torch.tensor(preprocessor.bounds[1], dtype=torch.float32, device=device),
    )

    def _train_member(member_seed):
        """Train one ensemble member (fresh init) and return it at its best
        validation checkpoint, plus epoch and history."""
        seed_everything(member_seed)
        model = PCPSMoE(
            static_dim=prepared.static.shape[1],
            sequence_channels=prepared.sequence.shape[2],
            config=model_config,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(train_config.epochs, 1), eta_min=5e-6)
        m_state, m_epoch, m_best, patience_left, m_hist = None, -1, float("inf"), train_config.patience, []
        for epoch in range(1, train_config.epochs + 1):
            train_diag = run_epoch(model, train_loader, optimizer, bounds, train_config, device)
            val_diag = run_epoch(model, val_loader, None, bounds, train_config, device)
            scheduler.step()
            m_hist.append({"epoch": epoch,
                           **{f"train_{k}": v for k, v in train_diag.items()},
                           **{f"val_{k}": v for k, v in val_diag.items()},
                           "learning_rate": optimizer.param_groups[0]["lr"]})
            cur_nll = float(val_diag["nll"])
            if cur_nll < m_best - 1e-4:
                m_best, m_epoch = cur_nll, epoch
                m_state = copy.deepcopy(model.state_dict())
                patience_left = train_config.patience
            else:
                patience_left -= 1
            if epoch == 1 or epoch % 10 == 0:
                print(f"{split.name} m{member_seed % 1000} epoch={epoch:03d} "
                      f"val_nll={val_diag['nll']:.4f}")
            if patience_left <= 0:
                break
        if m_state is None:
            raise RuntimeError("Training did not produce a valid checkpoint.")
        model.load_state_dict(m_state)
        return model, m_state, m_epoch, m_hist

    def _avg_predictions(pred_list):
        out = dict(pred_list[0])
        for key in ("prediction", "mixture_mean"):
            logs = [np.log(np.maximum(p[key], 1e-9)) for p in pred_list]
            out[key] = np.exp(np.mean(logs, axis=0))
        out["log_std"] = np.mean([p["log_std"] for p in pred_list], axis=0)
        return out

    # MULTI-SEED DEEP ENSEMBLE: train K members with different inits, average
    # their predictions (decorrelated -> genuine variance reduction).  K=1 is the
    # standard single model.  The recalibration below is applied to the averaged
    # prediction.
    n_members = max(1, int(getattr(train_config, "model_ensemble_k", 1)))
    val_list, test_list = [], []
    model = best_state = None
    best_epoch, history = -1, []
    for mi in range(n_members):
        model, best_state, best_epoch, history = _train_member(train_config.seed + 7919 * mi)
        val_list.append(predict(model, val_loader, device))
        test_list.append(predict(model, test_loader, device))
    val_pred = _avg_predictions(val_list)
    test_pred = _avg_predictions(test_list)
    model.load_state_dict(best_state)   # representative member for saving

    # Post-hoc per-target AFFINE recalibration in log-space, fitted on the
    # VALIDATION fold (never on test) and applied identically to val + test.
    # PC-PSMoE's regularized multiplicative-PKN-residual + heteroscedastic-NLL
    # head leaves the point prediction both mean-biased AND mis-scaled (the
    # log_y-vs-log_pred slope departs from 1); a 2-parameter calibration
    # log_y ~ a*log_pred + b corrects both.  This is standard held-out calibration
    # and is part of the proposed method's prediction pipeline (the point baselines
    # are vanilla).  `a` is clipped for stability.
    def _fit_affine(pred, y):
        lp = np.log(np.maximum(pred, 1e-5))
        ly = np.log(np.maximum(y, 1e-5))
        a = np.ones(pred.shape[1]); b = np.zeros(pred.shape[1])
        for j in range(pred.shape[1]):
            aj, bj = np.polyfit(lp[:, j], ly[:, j], 1)
            a[j], b[j] = float(np.clip(aj, 0.5, 1.5)), float(bj)
        return a[None, :], b[None, :]

    def _apply_affine(pred, a, b):
        return np.exp(a * np.log(np.maximum(pred, 1e-5)) + b)

    calib_a, calib_b = _fit_affine(val_pred["prediction"], val_pred["y"])
    val_pred["prediction"] = _apply_affine(val_pred["prediction"], calib_a, calib_b)
    test_pred["prediction"] = _apply_affine(test_pred["prediction"], calib_a, calib_b)

    val_log_error = np.abs(
        np.log(np.maximum(val_pred["y"], 1e-5))
        - np.log(np.maximum(val_pred["prediction"], 1e-5))
    )
    val_scale = np.maximum(val_pred["log_std"], 0.025)
    scores = val_log_error / val_scale
    n_val = len(scores)
    quantile_level = min(
        1.0, math.ceil((n_val + 1) * 0.90) / max(n_val, 1)
    )
    conformal_q = np.quantile(scores, quantile_level, axis=0, method="higher")
    test_log_prediction = np.log(np.maximum(test_pred["prediction"], 1e-5))
    radius = np.maximum(test_pred["log_std"], 0.025) * conformal_q
    lower = np.exp(test_log_prediction - radius)
    upper = np.exp(test_log_prediction + radius)
    lower = np.maximum(lower, preprocessor.bounds[0])
    upper = np.minimum(upper, preprocessor.bounds[1])

    y_true = test_pred["y"]
    y_hat = test_pred["prediction"]
    r2 = r2_score(y_true, y_hat, multioutput="raw_values")
    rmse = np.sqrt(mean_squared_error(y_true, y_hat, multioutput="raw_values"))
    mae = mean_absolute_error(y_true, y_hat, multioutput="raw_values")
    coverage = ((y_true >= lower) & (y_true <= upper)).mean(axis=0)
    interval_width = (upper - lower).mean(axis=0)
    gate_entropy = -np.sum(
        test_pred["gate_prob"]
        * np.log(np.maximum(test_pred["gate_prob"], 1e-9)),
        axis=1,
    )

    metric_row: dict[str, Any] = {
        "variant": ablation_config.variant,
        "fold": split.name,
        "n_train": len(split.train_idx),
        "n_val": len(split.val_idx),
        "n_test": len(split.test_idx),
        "best_epoch": best_epoch,
        "mean_r2": float(np.mean(r2)),
        "mean_rmse": float(np.mean(rmse)),
        "mean_mae": float(np.mean(mae)),
        "mean_gate_entropy": float(gate_entropy.mean()),
        "mean_max_gate_probability": float(
            test_pred["gate_prob"].max(axis=1).mean()
        ),
    }
    for i, target in enumerate(TARGETS):
        metric_row[f"{target}_r2"] = float(r2[i])
        metric_row[f"{target}_rmse"] = float(rmse[i])
        metric_row[f"{target}_mae"] = float(mae[i])
        metric_row[f"{target}_picp90"] = float(coverage[i])
        metric_row[f"{target}_mpiw90"] = float(interval_width[i])

    predictions = pd.DataFrame(
        {
            "sample_id": raw.sample_ids[split.test_idx],
            "well": raw.wells[split.test_idx],
            "stage": raw.stages[split.test_idx],
        }
    )
    for i, target in enumerate(TARGETS):
        predictions[f"true_{target}"] = y_true[:, i]
        predictions[f"pred_{target}"] = y_hat[:, i]
        predictions[f"lower90_{target}"] = lower[:, i]
        predictions[f"upper90_{target}"] = upper[:, i]
    for domain in range(model_config.n_domains):
        predictions[f"domain_probability_{domain + 1}"] = test_pred["gate_prob"][
            :, domain
        ]
    predictions.to_csv(
        fold_dir / "predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(history).to_csv(
        fold_dir / "training_history.csv", index=False, encoding="utf-8-sig"
    )
    np.save(fold_dir / "sequence_attention.npy", test_pred["sequence_attention"])
    torch.save(
        {
            "model_name": MODEL_NAME,
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "ablation_config": asdict(ablation_config),
            "state_dict": model.state_dict(),
            "static_dim": prepared.static.shape[1],
            "sequence_channels": prepared.sequence.shape[2],
            "conformal_q90": conformal_q,
        },
        fold_dir / "model.pt",
    )
    joblib.dump(preprocessor, fold_dir / "preprocessor.joblib")
    (fold_dir / "fold_manifest.json").write_text(
        json.dumps(
            {
                "model_name": MODEL_NAME,
                "abbreviation": MODEL_ABBR,
                "split": split.name,
                "train_indices": split.train_idx.tolist(),
                "validation_indices": split.val_idx.tolist(),
                "test_indices": split.test_idx.tolist(),
                "preprocessing": preprocessor.manifest(),
                "ablation": asdict(ablation_config),
                "metrics": metric_row,
                "conformal_q90": conformal_q.tolist(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"{split.name} complete: mean R2={metric_row['mean_r2']:.4f}, "
        f"best epoch={best_epoch}"
    )
    return metric_row


def random_splits(
    n_samples: int,
    repeats: int,
    seed: int,
) -> Iterable[FoldSplit]:
    all_idx = np.arange(n_samples)
    for repeat in range(repeats):
        split_seed = seed + 97 * repeat
        train_idx, holdout_idx = train_test_split(
            all_idx, test_size=0.40, random_state=split_seed
        )
        val_idx, test_idx = train_test_split(
            holdout_idx, test_size=0.50, random_state=split_seed + 1
        )
        yield FoldSplit(
            name=f"random_repeat_{repeat + 1:02d}",
            train_idx=np.sort(train_idx),
            val_idx=np.sort(val_idx),
            test_idx=np.sort(test_idx),
        )


def group_kfold_splits(
    wells: np.ndarray,
    n_splits: int,
    seed: int,
) -> Iterable[FoldSplit]:
    unique_wells = np.unique(wells)
    outer = GroupKFold(n_splits=min(n_splits, len(unique_wells)))
    all_idx = np.arange(len(wells))
    for fold, (fit_idx, test_idx) in enumerate(
        outer.split(all_idx, groups=wells), start=1
    ):
        inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed + fold)
        inner_train, inner_val = next(
            inner.split(fit_idx, groups=wells[fit_idx])
        )
        yield FoldSplit(
            name=f"groupkfold_{fold:02d}",
            train_idx=np.sort(fit_idx[inner_train]),
            val_idx=np.sort(fit_idx[inner_val]),
            test_idx=np.sort(test_idx),
        )


def lowo_splits(wells: np.ndarray, seed: int) -> Iterable[FoldSplit]:
    all_idx = np.arange(len(wells))
    for fold, heldout_well in enumerate(sorted(np.unique(wells)), start=1):
        test_idx = all_idx[wells == heldout_well]
        fit_idx = all_idx[wells != heldout_well]
        inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed + fold)
        inner_train, inner_val = next(
            inner.split(fit_idx, groups=wells[fit_idx])
        )
        safe_name = heldout_well.replace("/", "_").replace("\\", "_")
        yield FoldSplit(
            name=f"lowo_{fold:02d}_{safe_name}",
            train_idx=np.sort(fit_idx[inner_train]),
            val_idx=np.sort(fit_idx[inner_val]),
            test_idx=np.sort(test_idx),
        )


def temporal_split(
    wells: np.ndarray,
    operation_order: np.ndarray,
) -> FoldSplit:
    train_parts = []
    val_parts = []
    test_parts = []
    for well in sorted(np.unique(wells)):
        well_idx = np.where(wells == well)[0]
        ordered = well_idx[
            np.lexsort((well_idx, operation_order[well_idx]))
        ]
        n = len(ordered)
        train_end = max(1, int(math.floor(0.60 * n)))
        val_end = max(train_end + 1, int(math.floor(0.80 * n)))
        val_end = min(val_end, n - 1)
        train_parts.append(ordered[:train_end])
        val_parts.append(ordered[train_end:val_end])
        test_parts.append(ordered[val_end:])
    return FoldSplit(
        name="temporal_within_well_60_20_20",
        train_idx=np.sort(np.concatenate(train_parts)),
        val_idx=np.sort(np.concatenate(val_parts)),
        test_idx=np.sort(np.concatenate(test_parts)),
    )


def collect_splits(
    protocol: str,
    raw: RawData,
    repeats: int,
    group_folds: int,
    seed: int,
) -> list[FoldSplit]:
    if protocol == "random":
        return list(random_splits(len(raw.y), repeats, seed))
    if protocol == "groupkfold":
        return list(group_kfold_splits(raw.wells, group_folds, seed))
    if protocol == "lowo":
        return list(lowo_splits(raw.wells, seed))
    if protocol == "temporal":
        return [temporal_split(raw.wells, raw.operation_order)]
    return (
        list(random_splits(len(raw.y), repeats, seed))
        + list(group_kfold_splits(raw.wells, group_folds, seed))
        + list(lowo_splits(raw.wells, seed))
        + [temporal_split(raw.wells, raw.operation_order)]
    )


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in metrics.columns
        if column not in {"variant", "fold"}
        and pd.api.types.is_numeric_dtype(metrics[column])
    ]
    summary = metrics[metric_columns].agg(["mean", "std"]).T.reset_index()
    summary.columns = ["metric", "mean", "std"]
    return summary


def ablation_from_variant(variant: str) -> AblationConfig:
    mapping = {
        "full": AblationConfig("full"),
        "no_pkn": AblationConfig("no_pkn", use_pkn=False),
        "no_curve": AblationConfig("no_curve", use_curve=False),
        "no_static": AblationConfig("no_static", use_static=False),
        "no_physics": AblationConfig("no_physics", use_physics_losses=False),
        "no_group_dro": AblationConfig("no_group_dro"),
        "single_expert": AblationConfig("single_expert"),
    }
    return mapping[variant]


def apply_variant_to_configs(
    variant: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> AblationConfig:
    ablation_config = ablation_from_variant(variant)
    if variant == "no_physics":
        train_config.lambda_bounds = 0.0
        train_config.lambda_volume = 0.0
        train_config.lambda_monotonic = 0.0
    if variant == "no_group_dro":
        train_config.lambda_group_dro = 0.0
    if variant == "single_expert":
        model_config.n_domains = 1
        train_config.lambda_gate_balance = 0.0
        train_config.lambda_gate_entropy = 0.0
        train_config.lambda_expert_diversity = 0.0
    if variant == "no_static":
        # The no-static ablation should not receive operation quantities through
        # physics regularizers either.
        train_config.lambda_volume = 0.0
        train_config.lambda_monotonic = 0.0
    return ablation_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Train and validate the full {MODEL_ABBR} architecture."
    )
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "03_validation_results"
        / "pc_psmoe_full_rerun",
    )
    parser.add_argument(
        "--protocol",
        choices=["random", "groupkfold", "lowo", "temporal", "all"],
        default="random",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--group-folds", type=int, default=5)
    parser.add_argument(
        "--pkn-source",
        choices=["raw", "workbook-calibrated"],
        default="raw",
        help="Raw is reviewer-safe because calibration is fitted inside each fold.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.0)
    parser.add_argument("--gate-temperature", type=float, default=0.0)
    parser.add_argument("--target-gate-entropy-ratio", type=float, default=-1.0)
    parser.add_argument("--lambda-gate-entropy", type=float, default=-1.0)
    parser.add_argument("--lambda-expert-diversity", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument(
        "--variant",
        choices=[
            "full",
            "no_pkn",
            "no_curve",
            "no_static",
            "no_physics",
            "no_group_dro",
            "single_expert",
        ],
        default="full",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one tiny two-epoch fold to validate the complete code path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    model_config = ModelConfig()
    train_config = TrainConfig(seed=args.seed)
    if args.smoke_test:
        model_config = ModelConfig(
            static_hidden=32,
            sequence_width=32,
            pkn_hidden=16,
            fusion_width=64,
            expert_width=48,
            n_domains=3,
            transformer_layers=1,
            transformer_heads=4,
            dropout=0.05,
        )
        train_config.epochs = 2
        train_config.patience = 2
        train_config.batch_size = 96
        args.protocol = "random"
        args.repeats = 1
        args.max_folds = 1
    if args.epochs > 0:
        train_config.epochs = args.epochs
    if args.batch_size > 0:
        train_config.batch_size = args.batch_size
    if args.learning_rate > 0:
        train_config.learning_rate = args.learning_rate
    if args.gate_temperature > 0:
        model_config.gate_temperature = args.gate_temperature
    if args.target_gate_entropy_ratio >= 0:
        train_config.target_gate_entropy_ratio = args.target_gate_entropy_ratio
    if args.lambda_gate_entropy >= 0:
        train_config.lambda_gate_entropy = args.lambda_gate_entropy
    if args.lambda_expert_diversity >= 0:
        train_config.lambda_expert_diversity = args.lambda_expert_diversity
    ablation_config = apply_variant_to_configs(
        args.variant, model_config, train_config
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    print(f"Loading data from: {args.data_dir}")
    raw = load_raw_data(args.data_dir, pkn_source=args.pkn_source)
    splits = collect_splits(
        args.protocol,
        raw,
        repeats=args.repeats,
        group_folds=args.group_folds,
        seed=args.seed,
    )
    if args.max_folds > 0:
        splits = splits[: args.max_folds]
    print(
        f"{MODEL_ABBR}: samples={len(raw.y)}, sequence={raw.sequence_raw.shape[1:]}, "
        f"wells={len(np.unique(raw.wells))}, folds={len(splits)}, device={device}"
    )

    metric_rows = []
    for fold_number, split in enumerate(splits, start=1):
        print(f"\n=== Fold {fold_number}/{len(splits)}: {split.name} ===")
        fold_train_config = copy.deepcopy(train_config)
        fold_train_config.seed = args.seed + fold_number * 101
        metric_rows.append(
            fit_one_fold(
                raw,
                split,
                model_config,
                fold_train_config,
                ablation_config,
                args.output_dir,
                device,
            )
        )

    metrics = pd.DataFrame(metric_rows)
    summary = summarize_metrics(metrics)
    metrics.to_csv(
        args.output_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        args.output_dir / "metric_summary.csv", index=False, encoding="utf-8-sig"
    )
    run_manifest = {
        "model_name": MODEL_NAME,
        "abbreviation": MODEL_ABBR,
        "architecture": {
            "curve_branch": "full 120-step multiscale convolution plus Transformer attention encoder",
            "static_branch": "reduced direct geology/engineering variables; no well/stage identifiers or curve statistics",
            "physics_backbone": "fold-wise log-space PKN calibration followed by multiplicative residual correction",
            "gate": "task-aware differentiable soft fracturing-regime probabilities",
            "experts": "multi-target heteroscedastic probabilistic residual experts",
            "physics_losses": [
                "positivity by log-residual formulation",
                "training-fold engineering bounds",
                "weak fracture-volume consistency",
                "soft dose-volume monotonic consistency",
            ],
            "robustness": "well-group distributionally robust training; well identity is never an input feature",
            "uncertainty": "mixture aleatoric/domain uncertainty plus validation-fold conformal calibration",
        },
        "data_dir": str(args.data_dir),
        "protocol": args.protocol,
        "pkn_source": args.pkn_source,
        "variant": args.variant,
        "device": str(device),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "ablation_config": asdict(ablation_config),
        "folds": [split.name for split in splits],
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== PC-PSMoE summary ===")
    print(metrics[["fold", "mean_r2", "mean_rmse", "mean_mae"]].to_string(index=False))
    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
