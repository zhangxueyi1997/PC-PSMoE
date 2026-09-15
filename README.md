# PC-PSMoE

**Predicting Final Hydraulic Fracture Geometry: A Probabilistic Soft-Domain MoE Model
Constrained by Treatment Dynamics and Physics-Based PKN Priors**

Xueyi Zhang, Yu Li, Yu Wang, Dawei Deng, Tong Wu, Zhejun Pan (corresponding author)

Code for the manuscript, under review. PC-PSMoE predicts the final geometry of a fracturing
stage, length `L`, stage-level network width `W` and effective height `H` in meters, with a 90%
prediction interval on each target. The dataset is 1,967 stages from 55 horizontal wells across
four pads of the Gulong shale-oil area, Songliao Basin, China.

## Model

An in-fold-calibrated PKN-type scaling law supplies the scale anchor, and the network learns
multiplicative residuals in the log coordinate that anchor defines. Around it:

- a residual MLP encodes the static geological, geomechanical and completion record;
- a multi-scale convolutional Transformer encodes the executed treatment sequence
  (120 time steps × 8 channels);
- six heteroscedastic experts under soft-domain gating produce a joint lognormal-mixture
  prediction, trained with a well-level group-DRO objective;
- split-conformal calibration turns the mixture into 90% intervals.

Mean target R² is 0.704 under the random holdout and 0.696 under 55-fold leave-one-well-out
(LOWO) validation, first among 18 models from the physics-based, tree, deep-sequence and
physics-residual families. The calibrated PKN prior alone reaches 0.518 across the 55 held-out
wells. Over nine paired seeds the anchor is worth ΔR² = 0.031 ± 0.013 (p = 4.3×10⁻⁵), against
0.008 ± 0.017 (p = 0.09) for the same physics written into the loss as penalties.

## Data

`data/demo_sample/` holds a real eight-well subset, two wells per pad and 305 stages, with
identifiers recoded to pads A–D and sub-members Ⅰ–Ⅲ. It runs the full pipeline.

The complete 55-well dataset is not distributed. `data/field_records/` is the loader path for it
and expects `completion_geomechanics_parameters.xlsx` and `treatment_curve_matrix_120step.csv`.

## Quick start

```bash
pip install -r code/pc_psmoe_full_code/requirements_pc_psmoe.txt
```

Smoke test, about a minute on CPU:

```bash
python code/pc_psmoe_full_code/train_pc_psmoe_full.py --data-dir data/demo_sample --smoke-test --device cpu
```

Random-holdout run on the demo subset:

```bash
python code/pc_psmoe_full_code/train_pc_psmoe_full.py --data-dir data/demo_sample --protocol random --repeats 1 --device cpu
```

`run_all_comparisons.py` and `run_multiseed_ablation.py` read from `data/field_records/` and have
no `--data-dir` flag, so copy the demo subset into place first:

```bash
cp -r data/demo_sample data/field_records
python code/pc_psmoe_full_code/run_all_comparisons.py --quick --device cpu
```

`--variant` selects the ablations (`full`, `no_pkn`, `no_curve`, `no_static`, `no_physics`,
`no_group_dro`, `single_expert`) and `--protocol` the validation scheme (`random`, `groupkfold`,
`lowo`, `temporal`, `all`).

## Repository layout

```
code/pc_psmoe_full_code/
  train_pc_psmoe_full.py      model definition and training entry point
  run_all_comparisons.py      18-model dual-protocol benchmark and paired tests
  fair_baselines.py           matched-budget deep-sequence baselines (LSTM / TCN / Transformer / CNN1D)
  run_multiseed_ablation.py   paired multi-seed replication of the ablations
  aggregate_multiseed.py      aggregation and paired statistics over the multi-seed runs
  coverage_eval.py            empirical coverage (PICP) and width of the 90% intervals
  assemble_random_results.py  random-protocol result assembly
  quick_eval_random.py        quick evaluation helper
  auto_resume_lowo.py         resume helper for the 55-fold LOWO sweep
data/demo_sample/             real eight-well subset (305 stages)
```

`run_multiseed_ablation.py` writes to `results/multiseed/`, which `aggregate_multiseed.py` and
`coverage_eval.py` read. Its seed and variant lists are constants at the top of the file.

## Reproduction notes

- Preprocessing, PKN calibration, model selection and conformal quantiles are fitted inside each
  training fold.
- Main seed 20260610; multi-seed replication uses 20260610–20260615 and 20260617–20260619 and
  averages 0.693 ± 0.018.
- Conformal coverage under the random protocol is 0.901, 0.897 and 0.901 for L, W and H, with
  mean relative widths 0.42, 0.22 and 0.32.
- The prior is a PKN-type scaling model, given in full in Section 3.1 of the paper, and reaches
  the code as precomputed columns in the input workbook.
- CPU runs reproduce the reported numbers; `--device cuda` is available where a GPU is present.

## License

MIT, see [LICENSE](LICENSE).

## Citation

Citation information will be added once the manuscript is published.

---

## 中文说明

物理约束概率软域混合专家模型（PC-PSMoE），预测终态水力裂缝几何（缝长 L、段级缝网宽度 W、
有效高度 H），并给出 90% 保形预测区间。数据为松辽盆地古龙页岩油区块四个平台 55 口水平井的
1,967 段压裂记录。

做法是把逐折校准的 PKN 型尺度律当作对数坐标的原点，网络在该坐标下学习乘性残差；静态地质
工程记录经残差 MLP 编码，实际施工曲线（120 步 × 8 通道）经多尺度卷积 Transformer 编码，
六个异方差专家在软域门控下输出联合概率预测。随机留出与 55 折留一井（LOWO）两种协议下，
全目标平均 R² 分别为 0.704 与 0.696，在四类共 18 个模型中排名第一。

`data/demo_sample/` 是 8 口井 / 305 段的真实子集（每平台 2 口井，井号脱敏为平台 A–D、
小层 Ⅰ–Ⅲ），可跑通全部流程；完整 55 井数据不随代码发布。运行方式见上文。
