"""Bacteria/virus error analysis on out-of-fold predictions (no training, no GPU).

Example:  .venv\\Scripts\\python.exe analyze_errors.py --runs A2 M1 M2 M3 M4
Writes outputs/analysis/: bv_summary.json, bv_errors.csv, bv_confidence.png, bv_errors_grid.png.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from train import CACHE, CLASSES

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/analysis'
P = [f'p_{c}' for c in CLASSES]
B, V = 1, 2


def load(runs):
    d = [pd.read_csv(ROOT / 'outputs' / r / 'oof.csv').sort_values('new_filename').reset_index(drop=True) for r in runs]
    for x in d[1:]:
        assert (x.new_filename.values == d[0].new_filename.values).all() and (x.label.values == d[0].label.values).all()
    df = d[0][['new_filename', 'split', 'label', 'group', 'fold']].copy()
    df[P] = np.mean([x[P].values for x in d], 0)
    df['pred'] = df[P].values.argmax(1)
    # per-model argmax, to see whether errors are shared by all architectures
    votes = np.stack([x[P].values.argmax(1) for x in d], 1)
    df['n_models_correct'] = (votes == df.label.values[:, None]).sum(1)
    return df, len(runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', default=['A2', 'M1', 'M2', 'M3', 'M4'])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df, n = load(args.runs)
    stats = pd.read_csv(ROOT / 'outputs/eda/image_stats.csv')[['new_filename', 'width', 'height', 'aspect', 'mean', 'std']]
    df = df.merge(stats, on='new_filename', how='left')

    pn = df[df.label.isin([B, V])].copy()           # true pneumonia (bacteria or virus)
    pn['v_share'] = pn.p_virus / (pn.p_bacteria + pn.p_virus)   # virus share within b+v probability mass
    pn['err'] = pn.pred != pn.label
    pn['bv_err'] = pn.pred.isin([B, V]) & pn.err

    # 1) how much would the best possible b/v threshold help? (optimistic: tuned and scored on the same OOF)
    y = df.label.values
    base = f1_score(y, df.pred, labels=[0, 1, 2, 3], average='macro', zero_division=0)
    sweep = []
    for t in np.arange(0.30, 0.71, 0.02):
        p = df.pred.values.copy()
        m = np.isin(p, [B, V])
        vs = df.p_virus / (df.p_bacteria + df.p_virus)
        p[m] = np.where(vs[m] > t, V, B)
        sweep.append((round(float(t), 2), f1_score(y, p, labels=[0, 1, 2, 3], average='macro', zero_division=0)))
    best_t = max(sweep, key=lambda s: s[1])

    # 2) confidence and model agreement of b/v errors
    e = pn[pn.bv_err]
    conf = np.abs(e.v_share - 0.5) * 2               # 0 = coin flip, 1 = fully confident
    summary = dict(
        runs=args.runs, oof_macro_f1=base,
        bv_errors=int(len(e)), bacteria_as_virus=int((e.label == B).sum()), virus_as_bacteria=int((e.label == V).sum()),
        pneumonia_total=int(len(pn)),
        bv_error_confidence={'<0.2 (near 50/50)': int((conf < 0.2).sum()), '0.2-0.6': int(((conf >= 0.2) & (conf < 0.6)).sum()),
                             '>=0.6 (confident)': int((conf >= 0.6).sum())},
        bv_errors_all_models_wrong=int((e.n_models_correct == 0).sum()),
        bv_errors_some_model_right=int((e.n_models_correct > 0).sum()),
        threshold_sweep_best=dict(t=best_t[0], macro_f1=best_t[1], note='tuned and scored on the same OOF: optimistic'),
        threshold_sweep=sweep,
        error_rate_by_original_split=pn.groupby('split').err.mean().round(4).to_dict(),
        error_rate_by_fold=pn.groupby('fold').err.mean().round(4).to_dict(),
        # do errors differ in image properties? (compare means of wrong vs right, per true class)
        image_props_wrong_vs_right={c: pn[pn.label == i].groupby('err')[['width', 'aspect', 'mean', 'std']].mean().round(3).to_dict()
                                    for i, c in [(B, 'bacteria'), (V, 'virus')]},
    )
    (OUT / 'bv_summary.json').write_text(json.dumps(summary, indent=2, default=float), encoding='utf-8')
    e.assign(confidence=conf).sort_values('confidence', ascending=False)[
        ['new_filename', 'split', 'fold', 'label', 'pred', 'v_share', 'n_models_correct', 'confidence', 'width', 'aspect']
    ].to_csv(OUT / 'bv_errors.csv', index=False)

    # figure 1: virus-share histogram per true class
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for i, c in [(B, 'bacteria'), (V, 'virus')]:
        ax[0].hist(pn[pn.label == i].v_share, bins=40, alpha=0.6, label=f'true {c}')
    ax[0].axvline(0.5, color='k', lw=0.8)
    ax[0].set(xlabel='p(virus) / (p(bacteria)+p(virus))', ylabel='images', title='Ensemble OOF: bacteria vs virus')
    ax[0].legend()
    ts, fs = zip(*sweep)
    ax[1].plot(ts, fs, marker='o', ms=3)
    ax[1].axhline(base, color='gray', ls='--', lw=0.8, label='argmax')
    ax[1].set(xlabel='virus threshold', ylabel='OOF macro-F1', title='Threshold sweep (optimistic, same OOF)')
    ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / 'bv_confidence.png', dpi=130); plt.close(fig)

    # figure 2: most confident errors of each direction
    rows = [(B, 'bacteria -> virus'), (V, 'virus -> bacteria')]
    fig, ax = plt.subplots(2, 6, figsize=(15, 5.6))
    for r, (i, title) in enumerate(rows):
        top = e[e.label == i].assign(c=conf).sort_values('c', ascending=False).head(6)
        for j, (_, x) in enumerate(top.iterrows()):
            a = np.load(CACHE / f'{x.split}_images' / f'{x.new_filename}.npy')
            ax[r, j].imshow(a, cmap='gray'); ax[r, j].axis('off')
            ax[r, j].set_title(f'{title}\np_v={x.v_share:.2f} ok={x.n_models_correct}/{n}', fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / 'bv_errors_grid.png', dpi=110); plt.close(fig)

    print(json.dumps({k: v for k, v in summary.items() if k != 'threshold_sweep'}, indent=1, default=float))


if __name__ == '__main__':
    main()
