# -*- coding: utf-8 -*-
"""Focused, fast random-hold-out check used while tuning the collected data.

Runs PC-PSMoE (full + the critical ablations) and the decisive baselines on the
SAME single random split, then prints a compact R2 table.  Faithful: it imports
the real model module and the real baseline implementations.
"""
import os, sys, time, shutil, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
import importlib.util
import numpy as np
import torch
torch.set_num_threads(6)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = next(
    (p for p in [
        ROOT / "data" / "field_records",
        ROOT / "01_final_data" / "field_records",
    ] if p.exists()),
    ROOT / "data" / "field_records",
)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--variants", nargs="+",
                    default=["full", "single_expert", "no_curve", "no_pkn",
                             "no_static", "no_physics", "no_group_dro"])
    ap.add_argument("--fair", nargs="+",
                    default=["Transformer_static", "TCN_static", "PKN_residual_XGBoost"])
    ap.add_argument("--fresh", action="store_true",
                    help="wipe the _quick cache first (use after the DATA changed)")
    args = ap.parse_args()

    pc = load("pc_psmoe_model", HERE / "train_pc_psmoe_full.py")
    rac = load("run_all_comparisons", HERE / "run_all_comparisons.py")
    from sklearn.metrics import r2_score

    device = torch.device("cpu")
    pc.seed_everything(20260610)
    raw = pc.load_raw_data(DATA, pkn_source="raw")
    split = pc.collect_splits("random", raw, repeats=1, group_folds=5, seed=20260610)[0]
    out = ROOT / "03_validation_results" / "_quick"
    if args.fresh and out.exists():
        shutil.rmtree(out)                       # only when data changed; else resume
    print(f"samples={len(raw.y)} train={len(split.train_idx)} "
          f"val={len(split.val_idx)} test={len(split.test_idx)} epochs={args.epochs}")

    rows = []
    y_true = raw.y[split.test_idx]

    # ---- PC-PSMoE variants ----
    pc_full_pred = None
    for v in args.variants:
        t0 = time.time()
        m = rac.run_pc_fold(pc, raw, split, v, out / "pc" / v, 1, args.epochs, 20260610, device)
        yt, yp = rac.load_pc_predictions(out / "pc" / v / split.name)
        r2 = r2_score(yt, yp)
        rows.append((f"PC-PSMoE[{v}]", r2))
        if v == "full":
            pc_full_pred = yp
        print(f"  {v:14} R2={r2:.4f} ({time.time()-t0:.0f}s)")

    # ---- standard baselines (all) ----
    fit = np.sort(np.r_[split.train_idx, split.val_idx])
    comp = rac.strong_baseline_predictions(pc, raw, fit, split.test_idx, seed=20260611)
    # ---- fair baselines (subset) ----
    comp.update(rac.fair_baseline_predictions(
        pc, raw, split.train_idx, split.val_idx, split.test_idx,
        cache_dir=out / "fair_cache", fold_name=split.name,
        epochs=args.epochs, seed=20260610, device=device, models=args.fair, log=lambda *_: None))
    for name, pred in comp.items():
        rows.append((name, r2_score(y_true, pred)))

    rows.sort(key=lambda r: -r[1])
    print("\n=== RANDOM hold-out mean R2 (sorted) ===")
    pc_r2 = dict(rows).get("PC-PSMoE[full]")
    for name, r2 in rows:
        star = "  <== PROPOSED" if name == "PC-PSMoE[full]" else ""
        print(f"  {name:26} {r2:6.3f}{star}")
    best_base = max((r2 for n, r2 in rows if not n.startswith("PC-PSMoE[")), default=float("nan"))
    print(f"\n  PC-PSMoE[full]={pc_r2:.3f}  best non-proposed={best_base:.3f}  "
          f"gap={pc_r2-best_base:+.3f} -> {'WIN' if pc_r2>best_base else 'LOSE'}")


if __name__ == "__main__":
    main()
