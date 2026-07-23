# -*- coding: utf-8 -*-
"""90% prediction-interval empirical coverage (PICP) and width from multiseed runs."""
import sys, io, glob, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path
import numpy as np
import pandas as pd

MS = str(Path(__file__).resolve().parents[2] / 'results' / 'multiseed')
rows = []
targets = None
for f in sorted(glob.glob(os.path.join(MS, 'seed_*', 'full', 'random_repeat_01', 'predictions.csv'))):
    seed = f.split('seed_')[1].split(os.sep)[0]
    df = pd.read_csv(f)
    if targets is None:
        targets = [c[len('true_'):] for c in df.columns if c.startswith('true_')]
    r = {'seed': seed, 'n': len(df)}
    for t in targets:
        y = df[f'true_{t}'].to_numpy(float)
        lo = df[f'lower90_{t}'].to_numpy(float)
        hi = df[f'upper90_{t}'].to_numpy(float)
        r[f'picp_{t}'] = float(((y >= lo) & (y <= hi)).mean())
        r[f'relw_{t}'] = float(((hi - lo) / y).mean())   # width relative to observed value
    rows.append(r)
d = pd.DataFrame(rows)
print('随机留出协议（9种子，完整配置，n=394/种子）：')
for t in targets:
    print('  %-5s PICP = %.3f ± %.3f   平均相对宽度 = %.2f ± %.02f' % (
        t, d[f'picp_{t}'].mean(), d[f'picp_{t}'].std(ddof=1),
        d[f'relw_{t}'].mean(), d[f'relw_{t}'].std(ddof=1)))
print('  全目标合并 PICP = %.3f' % d[[f'picp_{t}' for t in targets]].mean().mean())
