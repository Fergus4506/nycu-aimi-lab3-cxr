"""Summary figures for the report (reads experiments.csv and OOF files; no training).

Example:  .venv\\Scripts\\python.exe report_figures.py
Writes outputs/report/: resolution_trend.png, experiments_oof.png, confusion_final.png, oof_vs_public.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from train import CLASSES

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/report'
BLUE, ORANGE, GRAY, INK, INK2 = '#2a78d6', '#eb6834', '#9a9893', '#0b0b0b', '#52514e'
SEEDS = ['A3_r18_768', 'S2_A3_seed1', 'S2_A3_seed2']   # ResNet18@768, 3 seeds (resolution plot)
FINAL = ['R4_geo_p12']    # final submission R4_geo_p12
P = [f'p_{c}' for c in CLASSES]

plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRAY, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': 0.6, 'axes.axisbelow': True})


def score(s):
    return float(str(s).split()[0]) if isinstance(s, str) and s.strip() and s[0] in '0~' else np.nan


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    e = pd.read_csv(ROOT / 'experiments.csv', dtype=str)
    e['oof'] = e.oof_macro_f1.astype(float)
    for i, rid in e.run_id.items():  # full-precision OOF so labels round like the report tables
        m = ROOT / 'outputs' / rid / 'metrics.json'
        if m.exists():
            e.loc[i, 'oof'] = json.loads(m.read_text())['oof_macro_f1']
    e['pub'] = e.kaggle.map(lambda s: score(str(s).lstrip('~')) if isinstance(s, str) else np.nan)
    e['std'] = e.fold_mean_std.str.split('±').str[1].astype(float)
    r = e.set_index('run_id')

    # 1) resolution trend, ResNet18 only (one axis, two series: OOF and public)
    res = [(224, 'B0'), (384, 'A1'), (512, 'A2'), (768, 'A3_r18_768'), (1024, 'A4_r18_1024')]
    x = [s for s, _ in res]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.errorbar(x, [r.oof[k] for _, k in res], yerr=[r['std'][k] for _, k in res], color=BLUE, lw=2, marker='o',
                ms=8, capsize=3, label='OOF macro-F1 (±fold std)')
    pub = [(s, r.pub[k]) for s, k in res if not np.isnan(r.pub[k])]
    ax.plot(*zip(*pub), color=ORANGE, lw=2, marker='s', ms=8, label='Kaggle public')
    seeds = [r.oof[k] for k in SEEDS]
    ax.scatter([768] * 3, seeds, color=BLUE, s=18, zorder=3, alpha=0.6)
    ax.annotate('3 seeds', (768, min(seeds)), textcoords='offset points', xytext=(6, -12), color=INK2, fontsize=8)
    ax.set(xticks=x, xlabel='input size (long edge, px)', ylabel='macro-F1', title='ResNet18: effect of resolution')
    ax.legend(frameon=False, loc='lower right')
    fig.tight_layout(); fig.savefig(OUT / 'resolution_trend.png', dpi=160, bbox_inches='tight'); plt.close(fig)

    # 2) all single-model experiments, sorted, B0 as reference
    single = e[~e.run_id.str.startswith('ENS')].sort_values('oof')
    fig, ax = plt.subplots(figsize=(7.2, 6.8))
    y = np.arange(len(single))
    ax.barh(y, single.oof - 0.78, left=0.78, height=0.62, color=BLUE, xerr=single['std'],
            error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
    ax.axvline(r.oof['B0'], color=GRAY, ls='--', lw=1)
    ax.text(r.oof['B0'], len(single) - 0.3, ' B0', color=INK2, fontsize=8)
    short = lambda m: m.split('.')[0].replace('_patch4_window7_224', '')
    ax.set_yticks(y, [f'{a}  ({short(m)}, {s}px, {l}, {g})' for a, m, s, l, g in
                      zip(single.run_id, single.model, single['size'], single.loss, single.augment)], fontsize=7.5)
    for yi, v in zip(y, single.oof):   # values in a fixed column right of every error bar
        ax.text(0.897, yi, f'{v:.3f}', va='center', ha='left', fontsize=7.5, color=INK)
    ax.set(xlim=(0.78, 0.896), xlabel='OOF macro-F1 (5-fold, error bar = fold std)', title='All single-model experiments')
    ax.grid(axis='y', visible=False)
    fig.tight_layout(); fig.savefig(OUT / 'experiments_oof.png', dpi=160, bbox_inches='tight'); plt.close(fig)

    # 3) confusion matrix of the final submission (OOF)
    d = [pd.read_csv(ROOT / 'outputs' / k / 'oof.csv').sort_values('new_filename') for k in FINAL]
    yt = d[0].label.values
    yp = np.mean([x[P].values for x in d], 0).argmax(1)
    cm = confusion_matrix(yt, yp, labels=[0, 1, 2, 3])
    rec = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(rec, cmap='Blues', vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{cm[i, j]}\n{rec[i, j]:.0%}', ha='center', va='center', fontsize=8,
                    color='white' if rec[i, j] > 0.55 else INK)
    ax.set(xticks=range(4), yticks=range(4), xticklabels=CLASSES, yticklabels=CLASSES, xlabel='predicted',
           ylabel='true', title='OOF confusion, final model R4 (R18@512, geo aug)')
    ax.grid(False)
    fig.tight_layout(); fig.savefig(OUT / 'confusion_final.png', dpi=160, bbox_inches='tight'); plt.close(fig)

    # 4) OOF vs public for every uploaded file
    k = e.dropna(subset=['pub'])
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.scatter(k.oof, k.pub, color=BLUE, s=40, zorder=3)
    KEY = {'B0', 'M7_effv2s', 'M6_cnx_in22k', 'M9_r34_768', 'A3_r18_768', 'R4_geo_p12', 'R8_sam', 'ENS3_R5_R8_R9', 'ENS5_A3'}
    for _, row in k[k.run_id.isin(KEY)].iterrows():  # label key points only; all values are in the report's table
        off = {'M9_r34_768': (-5, 5, 'right'), 'A3_r18_768': (-5, -9, 'right'), 'R5_geocolor_p12': (-4, 6, 'right'), 'A4_r18_1024': (-7, -3, 'right'), 'R4_geo_p12': (0, 8, 'center'), 'M6_cnx_in22k': (-6, 3, 'right'), 'ENS3_R5_R8_R9': (0, 7, 'center'), 'R9_sam_768': (0, 7, 'center'), 'ENS5_R18_A3x3_R5_R6': (0, -13, 'center'), 'R6_768_geocolor': (0, -13, 'center')}.get(row.run_id, (5, 3, 'left'))  # avoid overlaps
        name = {'R5_geocolor_p12': 'R5', 'R6_768_geocolor': 'R6', 'ENS5_R18_A3x3_R5_R6': 'ENS5_R18', 'ENS3_R5_R8_R9': 'ENS3_SAM', 'R9_sam_768': 'R9', 'A4_r18_1024': 'A4', 'R4_geo_p12': 'R4 (final)', 'M7_effv2s': 'M7'}.get(row.run_id, row.run_id)
        ax.annotate(name, (row.oof, row.pub), textcoords='offset points', xytext=off[:2], ha=off[2], fontsize=7.5, color=INK2)
    lo, hi = 0.81, 0.89
    ax.plot([lo, hi], [lo, hi], color=GRAY, lw=1, ls='--')
    ax.fill_between([lo, hi], [lo - 0.024, hi - 0.024], [lo + 0.024, hi + 0.024], color=GRAY, alpha=0.15, lw=0,
                    label='±0.024: 1 sd of macro-F1 on a 591-image subset')
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel='OOF macro-F1 (4,726 images)', ylabel='Kaggle public macro-F1',
           title='Local OOF vs public leaderboard')
    ax.legend(frameon=False, loc='upper left', fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / 'oof_vs_public.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    print('written:', sorted(p.name for p in OUT.glob('*.png')))
    print('final candidate OOF confusion (rows true):', cm.tolist())


if __name__ == '__main__':
    main()
