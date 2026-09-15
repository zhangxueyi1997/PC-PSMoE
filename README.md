# PC-PSMoE

**Predicting Final Hydraulic Fracture Geometry: A Probabilistic Soft-Domain MoE Model
Constrained by Treatment Dynamics and Physics-Based PKN Priors**

Xueyi Zhang, Yu Li, Yu Wang, Dawei Deng, Tong Wu, Zhejun Pan (corresponding author)

This repository holds the model, the baselines and the statistical analysis code behind the
manuscript (under review). PC-PSMoE estimates the final geometry of a hydraulic fracture —
length `L`, stage-level network width `W` and effective height `H`, all in meters — for a
single treatment stage, together with a calibrated 90% prediction interval for each target.

The study dataset is 1,967 stage-level records from 55 horizontal wells across four pads of
the Gulong shale-oil area, Songliao Basin, China.

## What the model does

An in-fold-calibrated PKN-type scaling law supplies a **scale anchor**, and the network learns
multiplicative residuals in the log coordinate defined by that anchor. The prior is therefore a
coordinate system, not a supervisory target and not a loss penalty. Around it:

- a residual MLP encodes the static geological, geomechanical and completion record;
- a multi-scale convolutional Transformer encodes the executed treatment sequence
  (120 time steps × 8 channels);
- six heteroscedastic experts under soft-domain gating produce a joint lognormal-mixture
  prediction, trained with a well-level group-DRO objective;
- split-conformal calibration converts the mixture into 90% intervals.

Headline results, reproduced by the scripts below: mean target R² of **0.704** under the random
holdout and **0.696** under 55-fold leave-one-well-out (LOWO) validation, first among 18 models
from the physics-based, tree, deep-sequence and physics-residual families. The calibrated PKN
prior on its own reaches 0.518 across the 55 held-out wells. Over nine paired seeds, moving the
prior into the output coordinate is worth ΔR² = 0.031 ± 0.013 (one-sided paired t, p = 4.3×10⁻⁵),
while injecting the same physics as loss penalties is not significant (0.008 ± 0.017, p = 0.09).

## Data availability

**No field data is distributed here.** Stage-level fracturing records, treatment sequences and
microseismic interpretation labels are subject to the operator's confidentiality terms. The
loader interface (`data/field_records/`) documents the expected input files:
`completion_geomechanics_parameters.xlsx` and `treatment_curve_matrix_120step.csv`.

`data/demo_sample/` is a **real** eight-well subset of the study dataset — two wells from each
of the four pads, 305 stages — released so that reviewers can run the whole pipeline end to end.
Its file structure is identical to the full dataset. Well and pad identifiers are recoded
(pads A–D, sub-members Ⅰ–Ⅲ) to match the desensitized naming used in the manuscript. The full
55-well dataset remains restricted; see the Data availability statement of the paper.

## Quick start

```bash
pip install -r code/pc_psmoe_full_code/requirements_pc_psmoe.txt
```

Smoke test on the demo subset (CPU, about one minute):

```bash
python code/pc_psmoe_full_code/train_pc_psmoe_full.py --data-dir data/demo_sample --smoke-test --device cpu
```

A full random-holdout run on the demo subset:

```bash
python code/pc_psmoe_full_code/train_pc_psmoe_full.py --data-dir data/demo_sample --protocol random --repeats 1 --device cpu
```

`run_all_comparisons.py` and `run_multiseed_ablation.py` resolve their input from
`data/field_records/` and take no `--data-dir` flag, so point them at the demo subset by copying
it into place first:

```bash
cp -r data/demo_sample data/field_records
python code/pc_psmoe_full_code/run_all_comparisons.py --quick --device cpu
```

Ablation variants of the main script are selected with `--variant`
(`full`, `no_pkn`, `no_curve`, `no_static`, `no_physics`, `no_group_dro`, `single_expert`) and
validation protocols with `--protocol` (`random`, `groupkfold`, `lowo`, `temporal`, `all`).

## Repository layout

```
code/pc_psmoe_full_code/
  train_pc_psmoe_full.py      model definition and training entry point (random / LOWO protocols, ablation variants)
  run_all_comparisons.py      18-model dual-protocol benchmark (incl. fair deep-sequence and PKN-residual tree baselines) and paired tests
  fair_baselines.py           matched-budget deep-sequence baselines (LSTM / TCN / Transformer / CNN1D)
  run_multiseed_ablation.py   paired multi-seed replication of the ablations
  aggregate_multiseed.py      aggregation and paired statistics over the multi-seed runs
  coverage_eval.py            empirical coverage (PICP) and width of the 90% conformal intervals
  assemble_random_results.py  random-protocol result assembly
  quick_eval_random.py        quick evaluation helper
  auto_resume_lowo.py         resume helper for the 55-fold LOWO sweep
data/demo_sample/             real eight-well subset (305 stages)
```

`run_multiseed_ablation.py` writes to `results/multiseed/`, which `aggregate_multiseed.py` and
`coverage_eval.py` then read. Its seed list and variant list are constants at the top of the file.

## Reproduction notes

- Preprocessing, PKN calibration, model selection and conformal quantiles are all fitted inside
  the training/validation fold, independently per fold. No test fold takes part in any fit.
- The main seed is 20260610. Multi-seed replication uses nine seeds: 20260610–20260615 and
  20260617–20260619. Over those seeds the full configuration averages 0.693 ± 0.018, and the
  0.704 quoted above is one of them.
- Conformal coverage under the random protocol is 0.901 ± 0.017, 0.897 ± 0.022 and 0.901 ± 0.013
  for L, W and H, with mean relative widths 0.42, 0.22 and 0.32.
- The physics prior is a PKN-*type* scaling model, not the standard PKN analytical solution.
  Section 3.1 of the paper gives the full form. The PKN columns are precomputed upstream and
  read from the input workbook.
- Runs are CPU-reproducible; `--device cuda` is available where a GPU is present.

## License

MIT, see [LICENSE](LICENSE).

## Citation

The manuscript is under review. Citation information will be added once it is published.

---

## 中文说明

物理约束概率软域混合专家模型（PC-PSMoE），用于终态水力裂缝几何（缝长 L、段级缝网宽度 W、
有效高度 H）的段级概率预测，并给出 90% 保形预测区间。研究数据为松辽盆地古龙页岩油区块
四个平台 55 口水平井的 1,967 段压裂记录。

核心做法是把逐折校准的 PKN 型尺度律当作**对数坐标的原点**（尺度锚点），网络在该坐标下学习
乘性残差；静态地质工程记录经残差 MLP 编码，实际施工曲线（120 步 × 8 通道）经多尺度卷积
Transformer 编码，六个异方差专家在软域门控下输出联合概率预测。随机留出与 55 折留一井
（LOWO）两种协议下，全目标平均 R² 分别为 0.704 与 0.696，在四类共 18 个模型中排名第一。

**本仓库不包含现场数据**；`data/demo_sample/` 提供 8 口井 / 305 段的真实子集（每个平台 2 口井，
井号已脱敏为平台 A–D、小层 Ⅰ–Ⅲ），可端到端跑通全部流程，完整 55 井数据仍受项目与作业方限制。
运行方式与目录说明见上文英文部分；论文投稿中，引用信息待补充。
