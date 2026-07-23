# -*- coding: utf-8 -*-
"""
多种子消融结果汇总
==================
运行方式（可随时运行，只统计已完成的组合）：
    python aggregate_multiseed.py

输出（写入 results/multiseed/）：
- multiseed_per_run.csv     每次训练的目标级与平均R2
- multiseed_summary.csv     各配置跨种子的平均R2（均值±标准差）
- multiseed_paired.csv      配对组件增益（full − variant）：均值、标准差、逐种子正负、配对t检验
- 控制台打印可直接写进论文5.3/5.5节的结论句
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE.parents[1] / "results" / "multiseed"
VARIANTS = ["full", "no_pkn", "no_physics", "single_expert"]
VARIANT_CN = {"full": "完整PC-PSMoE", "no_pkn": "移除PKN锚点",
              "no_physics": "移除物理惩罚项", "single_expert": "单概率专家"}


def r2(y: np.ndarray, yh: np.ndarray) -> float:
    ss = ((y - yh) ** 2).sum()
    tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ss / tot)


def load_run(pred_csv: Path) -> dict:
    df = pd.read_csv(pred_csv)
    targets = [c[len("true_"):] for c in df.columns if c.startswith("true_")]
    row = {}
    for t in targets:
        row[f"r2_{t}"] = r2(df[f"true_{t}"].to_numpy(float), df[f"pred_{t}"].to_numpy(float))
    row["mean_r2"] = float(np.mean([row[f"r2_{t}"] for t in targets]))
    return row


def main() -> None:
    rows = []
    for seed_dir in sorted(OUT_ROOT.glob("seed_*")):
        seed = int(seed_dir.name.split("_")[1])
        for variant in VARIANTS:
            pred = seed_dir / variant / "random_repeat_01" / "predictions.csv"
            if not pred.exists():
                continue
            row = {"seed": seed, "variant": variant}
            row.update(load_run(pred))
            rows.append(row)
    if not rows:
        print("尚无已完成的组合，请先运行 run_multiseed_ablation.py")
        sys.exit(0)

    per_run = pd.DataFrame(rows).sort_values(["seed", "variant"])
    per_run.to_csv(OUT_ROOT / "multiseed_per_run.csv", index=False, encoding="utf-8-sig")
    print("已完成组合：")
    print(per_run.pivot_table(index="seed", columns="variant", values="mean_r2").round(4).to_string())

    # ---- 各配置汇总 ----
    summ = (per_run.groupby("variant")["mean_r2"]
            .agg(["count", "mean", "std", "min", "max"]).reindex(VARIANTS))
    summ.to_csv(OUT_ROOT / "multiseed_summary.csv", encoding="utf-8-sig")
    print("\n各配置平均目标R2（跨种子）：")
    print(summ.round(4).to_string())

    # ---- 配对增益：仅使用四个变体齐全的种子 ----
    piv = per_run.pivot_table(index="seed", columns="variant", values="mean_r2")
    piv = piv.reindex(columns=VARIANTS)
    complete = piv.dropna(subset=VARIANTS)
    n = len(complete)
    print(f"\n变体齐全的种子数：{n}（配对统计基于这些种子）")
    if n == 0:
        print("尚无四个变体齐全的种子，配对统计跳过——继续运行 run_multiseed_ablation.py 补齐即可。")
        return
    paired_rows = []
    for variant in VARIANTS[1:]:
        delta = (complete["full"] - complete[variant]).to_numpy()
        rec = {"component": VARIANT_CN[variant], "n_seeds": n,
               "delta_mean": delta.mean(), "delta_std": delta.std(ddof=1) if n > 1 else np.nan,
               "delta_min": delta.min(), "delta_max": delta.max(),
               "n_positive": int((delta > 0).sum())}
        if HAVE_SCIPY and n >= 3:
            t = stats.ttest_rel(complete["full"], complete[variant], alternative="greater")
            rec["paired_t_p_onesided"] = float(t.pvalue)
        paired_rows.append(rec)
        # 目标级配对增益
        for tgt in [c[3:] for c in per_run.columns if c.startswith("r2_")]:
            pv = per_run.pivot_table(index="seed", columns="variant", values=f"r2_{tgt}")
            if not {"full", variant}.issubset(pv.columns):
                continue
            pv = pv.dropna(subset=["full", variant])
            if len(pv):
                rec[f"delta_{tgt}_mean"] = float((pv["full"] - pv[variant]).mean())
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(OUT_ROOT / "multiseed_paired.csv", index=False, encoding="utf-8-sig")
    print("\n配对组件增益（full − variant，平均目标R2）：")
    show = [c for c in paired.columns if not c.startswith("delta_r2_")]
    print(paired[show].round(4).to_string(index=False))

    # ---- 论文语句 ----
    print("\n" + "=" * 72)
    print("可直接更新论文的表述（5.3节与5.5节边界三）：")
    for _, r in paired.iterrows():
        p = f"，单侧配对t检验p={r['paired_t_p_onesided']:.3g}" if "paired_t_p_onesided" in r and pd.notna(r.get("paired_t_p_onesided")) else ""
        print(f"  {r['component']}：ΔR2 = {r['delta_mean']:.3f} ± {r['delta_std']:.3f}"
              f"（{r['n_seeds']}个种子中{r['n_positive']}个为正{p}）")
    full_stats = per_run[per_run.variant == "full"]["mean_r2"]
    print(f"  完整配置跨种子散布：{full_stats.mean():.3f} ± {full_stats.std(ddof=1):.3f}"
          f"（范围 {full_stats.min():.3f} - {full_stats.max():.3f}）"
          f" —— 可替换5.5节'一次重训练散布约0.04'的表述")


if __name__ == "__main__":
    main()
