"""Test-time augmentation: re-predict OOF (each fold's own val images) and the test set with fixed mild views.

Example:  .venv\\Scripts\\python.exe tta.py --runs A2 M1 M2 M3 M4
Writes outputs/<run>/oof_tta.csv (same columns as oof.csv) and outputs/<run>/test_probs_tta.npy (sample order),
then prints OOF macro-F1 without/with TTA per run and for the probability-mean ensemble.
Views: identity, scale 0.95 / 1.05, shift x +-3%. No flips (heart side would swap).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from train import CACHE, CLASSES, MEAN, STD, build_model, clahe, letterbox, loader, macro_f1

ROOT = Path(__file__).resolve().parent
P = [f'p_{c}' for c in CLASSES]
VIEWS = [(1.0, 0.0), (0.95, 0.0), (1.05, 0.0), (1.0, 0.03), (1.0, -0.03)]   # (scale, x shift as fraction)


class TTACXR(Dataset):
    def __init__(self, names, splits, size, use_clahe):
        self.names, self.splits, self.size, self.use_clahe = list(names), list(splits), size, use_clahe

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        a = np.load(CACHE / f'{self.splits[i]}_images' / f'{self.names[i]}.npy')
        if self.use_clahe:
            a = clahe(a)
        a = letterbox(a, self.size)
        out = []
        for s, dx in VIEWS:
            m = cv2.getRotationMatrix2D((self.size / 2, self.size / 2), 0, s)
            m[0, 2] += dx * self.size
            v = cv2.warpAffine(a, m, (self.size, self.size), flags=cv2.INTER_LINEAR, borderValue=0)
            out.append(((v.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1))
        return torch.from_numpy(np.stack(out)), 0


@torch.no_grad()
def predict_views(model, dl):
    model.eval()
    probs = []
    for x, _ in dl:
        b, t = x.shape[:2]
        x = x.flatten(0, 1).cuda(non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            p = model(x).float().softmax(1)
        probs.append(p.view(b, t, -1).cpu())
    return torch.cat(probs).numpy()          # N, views, classes


def run(run_id, test, workers):
    out = ROOT / 'outputs' / run_id
    oof = pd.read_csv(out / 'oof.csv')
    oof_tta, test_p = oof.copy(), []
    for ck_path in sorted(out.glob('fold*.pt')):
        ck = torch.load(ck_path, map_location='cpu', weights_only=False)
        assert ck['classes'] == CLASSES
        a, k = ck['args'], ck['fold']
        model = build_model(a['model'], a['size']); model.load_state_dict(ck['state_dict'])
        model = model.cuda().to(memory_format=torch.channels_last)
        bs = max(1, a['bs'] // len(VIEWS))
        args = argparse.Namespace(bs=bs, workers=workers)
        va = oof[oof.fold == k]
        pv = predict_views(model, loader(TTACXR(va.new_filename, va.split, a['size'], a.get('clahe', False)), args, False))
        # sanity: the identity view must reproduce the saved OOF probabilities (bf16 noise only)
        assert np.abs(pv[:, 0] - va[P].values).max() < 0.05, f'{ck_path}: identity view differs from oof.csv'
        oof_tta.loc[va.index, P] = pv.mean(1)
        pt = predict_views(model, loader(TTACXR(test.new_filename, ['test'] * len(test), a['size'], a.get('clahe', False)), args, False))
        test_p.append(pt.mean(1))
        print(f'{ck_path.relative_to(ROOT)} done', flush=True)
    oof_tta.to_csv(out / 'oof_tta.csv', index=False)
    np.save(out / 'test_probs_tta.npy', np.mean(test_p, 0))
    return oof, oof_tta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    test = pd.read_csv(ROOT / 'test_data_sample.csv')[['new_filename']]
    base, tta = [], []
    for r in args.runs:
        o, t = run(r, test, args.workers)
        o, t = o.sort_values('new_filename'), t.sort_values('new_filename')
        y = o.label.values
        print(f'{r}: OOF macro-F1 {macro_f1(y, o[P].values.argmax(1)):.4f} -> TTA {macro_f1(y, t[P].values.argmax(1)):.4f}', flush=True)
        base.append(o[P].values); tta.append(t[P].values)
    if len(args.runs) > 1:
        print(f'ensemble: {macro_f1(y, np.mean(base, 0).argmax(1)):.4f} -> TTA {macro_f1(y, np.mean(tta, 0).argmax(1)):.4f}')


if __name__ == '__main__':
    main()
