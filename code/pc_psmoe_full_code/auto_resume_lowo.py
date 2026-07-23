# -*- coding: utf-8 -*-
"""
Crash-proof, memory-bounded LOWO runner for run_all_comparisons.py.

Three layers of protection:
1. Crash recovery: non-zero exit (e.g. OpenMP conflict) triggers automatic
   restart after 30 s; per-fold cache means at most one in-progress fold is lost.
2. Bounded memory: FOLDS_PER_PROCESS causes run_all_comparisons.py to exit
   with code 75 after N folds (treated as planned recycle, restarted immediately),
   keeping process lifetime and memory growth bounded.
3. Python crash dumps (CrashDumps/*.dmp) are cleaned before each launch.

Environment variables mitigate torch/LightGBM duplicate OpenMP runtime conflicts.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE / "auto_resume.log"
RECYCLE_EXIT = 75

ENV = dict(
    os.environ,
    KMP_DUPLICATE_LIB_OK="TRUE",
    OMP_WAIT_POLICY="PASSIVE",
    OMP_NUM_THREADS="4",
    MKL_NUM_THREADS="4",
    FOLDS_PER_PROCESS="3",
)


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def clean_crash_dumps() -> None:
    dumps = Path.home() / "AppData" / "Local" / "CrashDumps"
    freed = 0
    for p in dumps.glob("python.exe.*.dmp"):
        try:
            freed += p.stat().st_size
            p.unlink()
        except OSError:
            pass
    if freed:
        log(f"cleaned {freed / 1e6:.0f} MB of python crash dumps")


def main() -> None:
    attempt = 0
    while True:
        attempt += 1
        clean_crash_dumps()
        log(f"attempt {attempt}: starting run_all_comparisons.py")
        ret = subprocess.run(
            [sys.executable, str(HERE / "run_all_comparisons.py")],
            cwd=str(HERE),
            env=ENV,
        ).returncode
        if ret == 0:
            log("COMPLETED successfully")
            break
        if ret == RECYCLE_EXIT:
            log("planned process recycle — restarting immediately")
            continue
        log(f"EXITED code={ret} — restarting in 30s")
        time.sleep(30)


if __name__ == "__main__":
    main()
