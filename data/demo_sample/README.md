# Sample subset (real data, 8 wells / 305 stages)

A real subset of the study dataset covering two wells from each of the four
pads (A–D), provided so that the complete pipeline can be executed end to
end. File structure is identical to the full dataset described in the paper:

- `completion_geomechanics_parameters.xlsx` — sheets `sample_index`,
  `static_features`, `pkn_and_targets`;
- `treatment_curve_matrix_120step.csv` — 120-step, 8-channel treatment
  sequences per stage.

The full 55-well dataset remains restricted by the project and operating
company; requests are handled per the Data availability statement of the paper.

Quick check:

```
python code/pc_psmoe_full_code/train_pc_psmoe_full.py --data-dir data/demo_sample --smoke-test --device cpu
```
