# PC-PSMoE: Physics-Constrained Probabilistic Soft-domain Mixture-of-Experts for Final Hydraulic Fracture Geometry Prediction

物理约束概率软域混合专家模型（PC-PSMoE）——面向终态水力裂缝几何（缝长 L、段级缝网宽度 W、有效高度 H）的段级概率预测。

本仓库包含论文《融合施工动态与校准PKN先验的概率软域混合专家模型用于终态水力裂缝几何预测》的全部模型、基线与统计分析代码。

## 数据可用性

**本仓库不包含任何现场数据。** 段级压裂记录、施工序列与微地震解释标签涉及作业方保密要求，
不随代码发布。代码中的数据加载接口（`data/field_records/`）说明了所需的
输入文件结构（`completion_geomechanics_parameters.xlsx` 与 `treatment_curve_matrix_120step.csv`）。

## 数据样例（真实子集，可直接跑通）

`data/demo_sample/` 提供研究数据集的一个真实子集：四个平台各取 2 口井，共 8 井 / 305 段，
文件结构与论文所述完整数据一致。该子集用于让评审者端到端运行全部代码；
完整 55 井数据仍受项目与作业方限制，获取方式见论文 Data availability 声明。

快速验证（CPU 约 1 分钟）：

```
python code/pc_psmoe_full_code/train_pc_psmoe_full.py --data-dir data/demo_sample --smoke-test --device cpu
```

## 环境

```
pip install -r code/pc_psmoe_full_code/requirements_pc_psmoe.txt
```

## 目录结构

```
code/pc_psmoe_full_code/
  train_pc_psmoe_full.py      模型定义与训练主脚本（随机留出 / 留一井 LOWO 协议、消融变体）
  run_all_comparisons.py      18 模型双协议基准（含公平深度时序与 PKN 残差树基线）与统计检验
  fair_baselines.py           公平深度时序基线（LSTM / TCN / Transformer / CNN1D）
  run_multiseed_ablation.py   多种子消融重复实验（锚点 / 惩罚项 / 单专家，配对设计）
  aggregate_multiseed.py      多种子结果聚合与配对统计
  coverage_eval.py            90% 保形预测区间的实测覆盖率（PICP）与宽度评估
  assemble_random_results.py  随机协议结果汇总
  quick_eval_random.py        快速评估工具
  auto_resume_lowo.py         LOWO 55 折断点续跑
```

## 复现要点

- 全部预处理、PKN 校准、模型选择与保形分位数均限制在训练折 / 验证折内（逐折独立拟合）。
- 主结果种子为 20260610；多种子重复使用 9 个种子（20260610–20260615、20260617–20260619）。
- 物理先验为 PKN 型尺度模型（非标准 PKN 解析解），完整口径见论文 3.1 节。

## 引用

论文投稿中，引用信息待补充。
