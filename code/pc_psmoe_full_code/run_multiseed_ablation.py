# -*- coding: utf-8 -*-
"""
多种子消融实验（论文补充实验：确证二阶组件增益的绝对幅度）
================================================================
运行方式：直接在 IDE 或终端执行
    python run_multiseed_ablation.py

- SEEDS 列表中的每个种子 × 4 个配置（full / no_pkn / no_physics / single_expert）
- 随机留出协议（60/20/20），同一种子下四个变体共享同一数据划分 → 组件差异为配对差异
- 断点续跑安全：已完成的 (种子, 变体) 组合会自动跳过，进程被杀后重新运行本脚本即可继续
- 运行期间阻止 Windows 休眠；每次训练的完整日志保存在输出目录的 train_log.txt
- 全部完成后自动调用 aggregate_multiseed.py 生成汇总表

预计耗时：CPU 上每次训练约 15-25 分钟，20 次共约 5-8 小时（可整夜运行）。
如果只想跑 3 个种子，把下面 SEEDS 列表删到前 3 个即可。
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ----------------------------- 配置 -----------------------------
#SEEDS = [20260610, 20260611, 20260613]
SEEDS = [20260615,20260617,20260618,20260619]
VARIANTS = ["full", "no_pkn", "no_physics", "single_expert"]
DEVICE = "cpu"          # 有可用 GPU 时改为 "cuda"

HERE = Path(__file__).resolve().parent
TRAIN_SCRIPT = HERE / "train_pc_psmoe_full.py"
DATA_DIR = HERE.parents[1] / "data" / "field_records"
OUT_ROOT = HERE.parents[1] / "results" / "multiseed"
# -----------------------------------------------------------------

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def keep_awake(enable: bool) -> None:
    if os.name == "nt":
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if enable else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)


def combo_done(out_dir: Path) -> bool:
    return (out_dir / "random_repeat_01" / "predictions.csv").exists()


def main() -> None:
    assert TRAIN_SCRIPT.exists(), TRAIN_SCRIPT
    assert (DATA_DIR / "completion_geomechanics_parameters.xlsx").exists(), DATA_DIR
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    combos = [(s, v) for s in SEEDS for v in VARIANTS]
    pending = [(s, v) for s, v in combos
               if not combo_done(OUT_ROOT / f"seed_{s}" / v)]
    print(f"[{datetime.now():%H:%M:%S}] 共 {len(combos)} 个组合，"
          f"已完成 {len(combos) - len(pending)}，待运行 {len(pending)}")

    env = dict(os.environ, PYTHONUTF8="1", KMP_DUPLICATE_LIB_OK="TRUE")
    keep_awake(True)
    durations: list[float] = []
    try:
        for k, (seed, variant) in enumerate(pending, 1):
            out_dir = OUT_ROOT / f"seed_{seed}" / variant
            out_dir.mkdir(parents=True, exist_ok=True)
            eta = ""
            if durations:
                remain = (len(pending) - k + 1) * sum(durations) / len(durations)
                eta = f"，预计剩余 {remain / 3600:.1f} 小时"
            print(f"[{datetime.now():%H:%M:%S}] ({k}/{len(pending)}) "
                  f"seed={seed} variant={variant} 开始{eta}", flush=True)
            t0 = time.time()
            cmd = [sys.executable, str(TRAIN_SCRIPT),
                   "--protocol", "random", "--repeats", "1",
                   "--seed", str(seed), "--variant", variant,
                   "--device", DEVICE,
                   "--data-dir", str(DATA_DIR),
                   "--output-dir", str(out_dir)]
            with open(out_dir / "train_log.txt", "w", encoding="utf-8") as log:
                ret = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                     env=env).returncode
            dt = time.time() - t0
            if ret != 0 or not combo_done(out_dir):
                print(f"    !! 失败（退出码 {ret}），日志见 {out_dir / 'train_log.txt'}；"
                      f"继续下一个组合", flush=True)
                continue
            durations.append(dt)
            print(f"    完成，用时 {dt / 60:.1f} 分钟", flush=True)
    finally:
        keep_awake(False)

    print(f"[{datetime.now():%H:%M:%S}] 全部组合处理完毕，开始汇总……")
    subprocess.run([sys.executable, str(HERE / "aggregate_multiseed.py")], env=env)


if __name__ == "__main__":
    main()
