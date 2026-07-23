# -*- coding: utf-8 -*-
"""Assemble the random-split R2 table from saved PC-PSMoE predictions + (re)computed
baselines, without re-running the slow PC training."""
import os, sys, importlib.util
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
import numpy as np, pandas as pd, torch
torch.set_num_threads(6)
from sklearn.metrics import r2_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = next(
    (p for p in [
        ROOT / "data" / "field_records",
        ROOT / "01_final_data" / "field_records",
    ] if p.exists()),
    ROOT / "data" / "field_records",
)
QUICK = ROOT / "03_validation_results" / "_quick"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


pc = load("pc_psmoe_model", HERE / "train_pc_psmoe_full.py")
rac = load("run_all_comparisons", HERE / "run_all_comparisons.py")

pc.seed_everything(20260610)
raw = pc.load_raw_data(DATA, pkn_source="raw")
split = pc.collect_splits("random", raw, repeats=1, group_folds=5, seed=20260610)[0]
y_true = raw.y[split.test_idx]
fit = np.sort(np.r_[split.train_idx, split.val_idx])

rows = []
# PC variants from saved predictions
for v in ["full", "single_expert", "no_curve", "no_pkn", "no_static", "no_physics", "no_group_dro"]:
    p = QUICK / "pc" / v / split.name / "predictions.csv"
    if p.exists():
        df = pd.read_csv(p)
        yt = df[[f"true_{t}" for t in pc.TARGETS]].to_numpy(float)
        yp = df[[f"pred_{t}" for t in pc.TARGETS]].to_numpy(float)
        rows.append((f"PC-PSMoE[{v}]", r2_score(yt, yp)))

# standard baselines (fast)
comp = rac.strong_baseline_predictions(pc, raw, fit, split.test_idx, seed=20260611)
# PKN-residual XGBoost (fast) — same seed scheme as fair_baseline_predictions (i=2)
yhat = rac.pkn_residual_tree(pc, raw, fit, split.test_idx, seed=20260610 + 2000, kind="xgb")
if yhat is not None:
    comp["PKN_residual_XGBoost"] = yhat
# cached deep baselines if present
for nm in ["Transformer_static", "TCN_static"]:
    f = QUICK / "fair_cache" / split.name / f"{nm}.npy"
    if f.exists():
        comp[nm] = np.load(f)
for nm, pred in comp.items():
    rows.append((nm, r2_score(y_true, pred)))

rows.sort(key=lambda r: -r[1])
print("\n=== RANDOM hold-out mean R2 (sorted) ===")
d = dict(rows); pc_r2 = d.get("PC-PSMoE[full]")
for name, r2 in rows:
    star = "  <== PROPOSED" if name == "PC-PSMoE[full]" else ""
    print(f"  {name:26} {r2:6.3f}{star}")
best = max(r2 for n, r2 in rows if not n.startswith("PC-PSMoE["))
print(f"\n  full={pc_r2:.3f}  best non-proposed={best:.3f}  gap={pc_r2-best:+.3f} "
      f"-> {'WIN' if pc_r2>best else 'LOSE'}")
print("\n  --- ablation drops (full - variant; want all > 0) ---")
for v in ["single_expert", "no_curve", "no_pkn", "no_static", "no_physics", "no_group_dro"]:
    if f"PC-PSMoE[{v}]" in d:
        print(f"    {v:14} drop = {pc_r2 - d[f'PC-PSMoE[{v}]']:+.3f}")
