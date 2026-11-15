"""Average fold models of one or more runs on the test set and write a checked submission CSV.

Example:  .venv\\Scripts\\python.exe predict.py --runs B0 --name B0
Writes submissions/<name>.csv and outputs/<run>/test_probs.npy (per-run average, sample order).
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train import CLASSES, CXR, build_model, loader, predict

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / 'test_data_sample.csv'
COLUMNS = ['new_filename', 'normal', 'bacteria', 'virus', 'COVID-19']


def check_submission(path):
    sub, sample = pd.read_csv(path), pd.read_csv(SAMPLE)
    assert list(sub.columns) == COLUMNS, f'columns {list(sub.columns)}'
    assert len(sub) == len(sample) == 1182, f'rows {len(sub)}'
    assert (sub.new_filename.values == sample.new_filename.values).all(), 'filename order differs from sample'
    v = sub[COLUMNS[1:]]
    assert v.notna().all().all() and v.isin([0, 1]).all().all(), 'values must be 0/1'
    assert (v.sum(axis=1) == 1).all(), 'each row needs exactly one 1'


def run_probs(run, test, args):
    out = ROOT / 'outputs' / run
    ckpts = sorted(out.glob('fold*.pt'))
    assert ckpts, f'no checkpoints in {out}'
    cached = out / 'test_probs.npy'
    if cached.exists() and cached.stat().st_mtime > max(c.stat().st_mtime for c in ckpts):
        print(f'{cached.relative_to(ROOT)}: reused', flush=True)   # only if newer than every checkpoint
        return np.load(cached)
    probs = []
    for c in ckpts:
        ck = torch.load(c, map_location='cpu', weights_only=False)
        assert ck['classes'] == CLASSES, f'{c}: class order mismatch'
        a = ck['args']
        model = build_model(a['model'], a['size'])
        model.load_state_dict(ck['state_dict'])
        model = model.cuda().to(memory_format=torch.channels_last)
        dl = loader(CXR(test.new_filename, ['test'] * len(test), [0] * len(test), a['size'], 'none', a.get('clahe', False),
                        a.get('norm', False)), argparse.Namespace(bs=a['bs'], workers=args.workers), False)
        probs.append(predict(model, dl, a.get('hflip', False)))
        print(f'{c.relative_to(ROOT)}: done', flush=True)
    p = np.mean(probs, 0)
    np.save(out / 'test_probs.npy', p)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    test = pd.read_csv(SAMPLE)[['new_filename']]
    p = np.mean([run_probs(r, test, args) for r in args.runs], 0)
    pred = p.argmax(1)
    sub = test.copy()
    for i, c in enumerate(CLASSES):  # write by class name, not by column position
        sub[c] = (pred == i).astype(int)
    sub = sub[COLUMNS]
    path = ROOT / 'submissions' / f'{args.name}.csv'
    path.parent.mkdir(exist_ok=True)
    assert not path.exists(), f'{path} exists; submissions are never overwritten'
    sub.to_csv(path, index=False)
    check_submission(path)
    print(f'{path.relative_to(ROOT)} ok  sha256={hashlib.sha256(path.read_bytes()).hexdigest()[:16]}')
    print('predicted counts:', dict(zip(CLASSES, np.bincount(pred, minlength=4).tolist())))


if __name__ == '__main__':
    main()
