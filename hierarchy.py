"""Hierarchical combination: 4-class model(s) decide pneumonia vs not; a bacteria/virus specialist splits pneumonia.

p_b' = (p_b + p_v) * q_b,  p_v' = (p_b + p_v) * q_v,  q = specialist softmax restricted to [bacteria, virus].
Example:  .venv\\Scripts\\python.exe hierarchy.py --base A2 M1 M2 M3 M4 --spec H1_bv_r18
          add --name H_ENS5 to also write submissions/H_ENS5.csv from test_probs.npy of every run.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from predict import COLUMNS, check_submission, run_probs
from train import CLASSES

ROOT = Path(__file__).resolve().parent
P = [f'p_{c}' for c in CLASSES]
B, V = 1, 2


def combine(base, spec, w=1.0):
    """w=1: specialist alone splits b/v; w=0.5: average of specialist and base split."""
    out = base.copy()
    pn = base[:, B] + base[:, V]
    q = spec[:, [B, V]] / spec[:, [B, V]].sum(1, keepdims=True)
    q0 = base[:, [B, V]] / np.clip(pn[:, None], 1e-12, None)
    q = w * q + (1 - w) * q0
    out[:, B], out[:, V] = pn * q[:, 0], pn * q[:, 1]
    return out


def load_oof(run):
    return pd.read_csv(ROOT / 'outputs' / run / 'oof.csv').sort_values('new_filename').reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', nargs='+', required=True)
    ap.add_argument('--spec', nargs='+', required=True)
    ap.add_argument('--w', type=float, default=1.0, help='weight of the specialist in the b/v split (fixed, not tuned)')
    ap.add_argument('--name', default='')
    args = ap.parse_args()

    base = [load_oof(r) for r in args.base]
    spec = [load_oof(r) for r in args.spec]
    for d in base[1:] + spec:
        assert (d.new_filename.values == base[0].new_filename.values).all()
    y = base[0].label.values
    pb, ps = np.mean([d[P].values for d in base], 0), np.mean([d[P].values for d in spec], 0)
    f = lambda p: f1_score(y, p.argmax(1), labels=[0, 1, 2, 3], average=None, zero_division=0)
    pn =np.isin(y, [B, V])
    bv = lambda p: f1_score(y[pn], np.where(p[pn][:, B] >= p[pn][:, V], B, V), labels=[B, V], average='macro')
    for name, p in [('base', pb), (f'hierarchy w={args.w}', combine(pb, ps, args.w))]:
        s = f(p)
        print(f'{name:16s} macro-F1 {s.mean():.4f}  per-class {np.round(s, 4).tolist()}  b/v-only macro-F1 {bv(p):.4f}')
    print(f'specialist alone on true b/v images: b/v macro-F1 {bv(ps):.4f}')

    if args.name:
        test = pd.read_csv(ROOT / 'test_data_sample.csv')[['new_filename']]
        def probs(r):  # reuse cached test probabilities; compute them once if missing
            f = ROOT / 'outputs' / r / 'test_probs.npy'
            return np.load(f) if f.exists() else run_probs(r, test, argparse.Namespace(workers=4))
        tb = np.mean([probs(r) for r in args.base], 0)
        ts = np.mean([probs(r) for r in args.spec], 0)
        pred = combine(tb, ts, args.w).argmax(1)
        sub = test.copy()
        for i, c in enumerate(CLASSES):
            sub[c] = (pred == i).astype(int)
        path = ROOT / 'submissions' / f'{args.name}.csv'
        assert not path.exists(), f'{path} exists'
        sub[COLUMNS].to_csv(path, index=False)
        check_submission(path)
        print(path.relative_to(ROOT), 'ok', dict(zip(CLASSES, np.bincount(pred, minlength=4).tolist())))


if __name__ == '__main__':
    main()
