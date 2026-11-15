"""5-fold training with out-of-fold (OOF) evaluation for Lab 3.

Example (B0):  .venv\\Scripts\\python.exe train.py --run B0 --model resnet18 --size 224
Outputs go to outputs/<run>/ and one row is appended to experiments.csv.
"""
import argparse
import csv
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'data/cache/long512_npy'
CLASSES = ['normal', 'bacteria', 'virus', 'COVID-19']   # index = model output index
CACHES = [512, 768, 1024]   # long-edge caches built by prepare_input.py; never upsample a smaller cache


def cache_for(size):
    edge = next(e for e in CACHES if size <= e)
    # LAB3_CACHE_ROOT: an SSD copy of data/cache (D: is an HDD; random reads there throttle 768px training ~3x)
    return Path(os.environ.get('LAB3_CACHE_ROOT', ROOT / 'data/cache')) / f'long{edge}_npy'


MEAN, STD =np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def letterbox(a, size):
    """Keep aspect ratio, resize long edge to `size`, zero-pad to a square."""
    h, w = a.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    a = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    out = np.zeros((size, size, 3), a.dtype)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = a
    return out


def clahe(a):
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY))
    return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)


def intensity_norm(a):
    """Per-image window: clip to the 0.5-99.5 percentiles and stretch to 0-255 (sources differ in brightness)."""
    lo, hi = np.percentile(a, (0.5, 99.5))
    return np.clip((a.astype(np.float32) - lo) * (255 / max(hi - lo, 1.0)), 0, 255).astype(np.uint8)


def color_jitter(a, rng):
    """Gain x0.85-1.15, offset +-19 grey levels; applied before letterbox so the padding stays black."""
    return np.clip(a.astype(np.float32) * rng.uniform(0.85, 1.15) + rng.uniform(-19, 19), 0, 255).astype(np.uint8)


def geometric(a, rng):
    """Rotate +-10 deg, scale 0.9-1.1, shift +-5% on the letterboxed square."""
    size = a.shape[0]
    m = cv2.getRotationMatrix2D((size / 2, size / 2), rng.uniform(-10, 10), rng.uniform(0.9, 1.1))
    m[:, 2] += rng.uniform(-0.05, 0.05, 2) * size
    return cv2.warpAffine(a, m, (size, size), flags=cv2.INTER_LINEAR, borderValue=0)


class CXR(Dataset):
    def __init__(self, names, splits, labels, size, aug='none', use_clahe=False, norm=False, hflip=False):
        self.names, self.splits, self.labels, self.size = list(names), list(splits), list(labels), size
        self.aug, self.use_clahe, self.norm, self.hflip = aug, use_clahe, norm, hflip

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        a = np.load(cache_for(self.size) / f'{self.splits[i]}_images' / f'{self.names[i]}.npy')
        if self.use_clahe:
            a = clahe(a)
        if self.norm:
            a = intensity_norm(a)
        rng = np.random.default_rng()
        if self.hflip and rng.random() < 0.5:  # train only; b/v cues are symmetry/pattern, not side
            a = a[:, ::-1]
        if self.aug == 'geo_color':
            a = color_jitter(a, rng)
        a = letterbox(a, self.size)
        if self.aug != 'none':
            a = geometric(a, rng)
        a = (a.astype(np.float32) / 255 - MEAN) / STD
        return torch.from_numpy(a.transpose(2, 0, 1).copy()), self.labels[i]


def worker_init(_):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, y):
        logp = logits.log_softmax(1).gather(1, y[:, None]).squeeze(1)
        return (-(1 - logp.exp()) ** self.gamma * logp).mean()


def make_loss(name, counts):
    if name == 'wce':  # N / (C * n_c), from this fold's training labels only
        w = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)
        return torch.nn.CrossEntropyLoss(weight=w.cuda())
    if name == 'focal':
        return FocalLoss(2.0)
    if name == 'ls':
        return torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    return torch.nn.CrossEntropyLoss()


def loader(ds, args, shuffle, sampler=None):
    return DataLoader(ds, batch_size=args.bs, shuffle=shuffle and sampler is None, sampler=sampler,
                      num_workers=args.workers, pin_memory=True,
                      persistent_workers=args.workers > 0, worker_init_fn=worker_init, drop_last=False,
                      **({'multiprocessing_context': 'spawn'} if args.workers else {}))


def build_model(name, size=None):
    if '.' in name:  # timm name with pretrained tag, e.g. convnext_tiny.fb_in22k_ft_in1k (ImageNet weights only)
        import timm
        kw = {'img_size': size} if name.startswith('swin') else {}   # Swin needs the input size at build time
        return timm.create_model(name, pretrained=True, num_classes=len(CLASSES), **kw)
    m = torchvision.models.get_model(name, weights='DEFAULT')
    if hasattr(m, 'fc'):
        m.fc = torch.nn.Linear(m.fc.in_features, len(CLASSES))
    else:  # densenet / efficientnet / convnext style heads
        head = m.classifier
        last = head[-1] if isinstance(head, torch.nn.Sequential) else head
        new = torch.nn.Linear(last.in_features, len(CLASSES))
        if isinstance(head, torch.nn.Sequential):
            head[-1] = new
        else:
            m.classifier = new
    return m


@torch.no_grad()
def predict(model, dl, flip_tta=False):
    model.eval()
    probs = []
    for x, _ in dl:
        x = x.cuda(non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            p = model(x).float().softmax(1)
            if flip_tta:  # average with the mirrored image (models trained with --hflip)
                p = (p + model(x.flip(3)).float().softmax(1)) / 2
        probs.append(p.cpu())
    return torch.cat(probs).numpy()


def macro_f1(y, p):
    return f1_score(y, p, labels=[0, 1, 2, 3], average='macro', zero_division=0)


def run_fold(k, folds, args, out):
    tr, va = folds[folds.fold != k], folds[folds.fold == k]
    if args.limit:  # smoke test: tiny stratified subset
        tr = tr.groupby('label').head(args.limit)
        va = va.groupby('label').head(max(2, args.limit // 4))
    keep = [CLASSES.index(c) for c in args.only.split(',')] if args.only else None
    if keep:  # specialist: train on these classes only, still predict every val image (for hierarchical use)
        tr = tr[tr.label.isin(keep)]
    counts = np.bincount(tr.label, minlength=len(CLASSES))
    sampler = None
    if args.sampler:  # class-balanced sampling with replacement, same epoch length
        w = 1.0 / counts[tr.label.values]
        sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(w, dtype=torch.double), len(tr), replacement=True)
    dl_tr = loader(CXR(tr.new_filename, tr.split, tr.label, args.size, args.aug, args.clahe, args.norm, args.hflip),
                   args, True, sampler)
    dl_va = loader(CXR(va.new_filename, va.split, va.label, args.size, 'none', args.clahe, args.norm), args, False)

    model = build_model(args.model, args.size)
    if args.init:  # FixRes-style fine-tune: start from the same fold's model of another run (never saw this fold)
        ck = torch.load(ROOT / 'outputs' / args.init / f'fold{k}.pt', map_location='cpu', weights_only=False)
        assert ck['fold'] == k and ck['args']['model'] == args.model, f'{args.init} fold{k}: mismatch'
        model.load_state_dict(ck['state_dict'])
    model = model.cuda().to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total, warm = args.epochs * len(dl_tr), args.warmup * len(dl_tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    loss_fn = make_loss(args.loss, counts)
    # EMA of weights (and BN buffers); evaluated and saved instead of the raw model
    ema = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema), use_buffers=True) if args.ema else None
    eval_model = ema.module if ema else model

    best, best_ep, best_probs, history, bad = -1, -1, None, [], 0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run_loss, n = time.time(), 0.0, 0
        for x, y in dl_tr:
            x = x.cuda(non_blocking=True).to(memory_format=torch.channels_last)
            y = y.cuda(non_blocking=True)
            if args.mixup:  # mix pairs of images; loss mixes both labels with the same weight
                lam = float(np.random.beta(args.mixup, args.mixup))
                perm = torch.randperm(len(y), device=y.device)
                x = lam * x + (1 - lam) * x[perm]
            def step_loss():
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(x).float()
                    return lam * loss_fn(logits, y) + (1 - lam) * loss_fn(logits, y[perm]) if args.mixup else loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss = step_loss()
            loss.backward()
            if args.sam:  # SAM (Foret et al. 2021): take the gradient at w + rho*g/|g|, then step from w
                params = [p for p in model.parameters() if p.grad is not None]
                with torch.no_grad():
                    norm = torch.norm(torch.stack([p.grad.norm(2) for p in params]), 2) + 1e-12
                    eps = [p.grad * (args.sam / norm) for p in params]
                    for p, e in zip(params, eps):
                        p.add_(e)
                opt.zero_grad(set_to_none=True)
                bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
                moms = [m.momentum for m in bns]
                for m in bns:  # BN running stats are updated by the first pass only
                    m.momentum = 0.0
                step_loss().backward()
                for m, mo in zip(bns, moms):
                    m.momentum = mo
                with torch.no_grad():
                    for p, e in zip(params, eps):
                        p.sub_(e)
            opt.step()
            sched.step()
            if ema:
                ema.update_parameters(model)
            run_loss += loss.item() * len(y)
            n += len(y)
        probs = predict(eval_model, dl_va, args.hflip)
        m = va.label.isin(keep).values if keep else np.ones(len(va), bool)   # score only the classes trained on
        val_loss = float(torch.nn.functional.nll_loss(torch.log(torch.from_numpy(probs[m]).clamp_min(1e-8)),
                                                      torch.tensor(va.label.values[m])).item())
        if keep:
            f1 = f1_score(va.label.values[m], np.array(keep)[probs[m][:, keep].argmax(1)], labels=keep, average='macro')
        else:
            f1 = macro_f1(va.label.values, probs.argmax(1))
        history.append(dict(fold=k, epoch=ep, train_loss=run_loss / n, val_loss=val_loss, val_macro_f1=f1,
                            lr=opt.param_groups[0]['lr'], seconds=time.time() - t0))
        print(f'fold {k} ep {ep:2d} train_loss {run_loss / n:.4f} val_loss {val_loss:.4f} '
              f'val_f1 {f1:.4f} {time.time() - t0:.1f}s', flush=True)
        if f1 > best:
            best, best_ep, best_probs, bad = f1, ep, probs, 0
            torch.save({'state_dict': eval_model.state_dict(), 'classes': CLASSES, 'args': vars(args), 'fold': k,
                        'epoch': ep, 'val_macro_f1': f1}, out / f'fold{k}.pt')
        else:
            bad += 1
            if bad >= args.patience:
                break
    return va, best_probs, best, best_ep, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--model', default='resnet18', help='torchvision model name')
    ap.add_argument('--size', type=int, default=224)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=7)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--warmup', type=int, default=1, help='warmup epochs')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0, help='smoke test: N train images per class')
    ap.add_argument('--aug', default='none', choices=['none', 'geo', 'geo_color'])
    ap.add_argument('--clahe', action='store_true', help='CLAHE on all splits (also applied by predict.py)')
    ap.add_argument('--loss', default='ce', choices=['ce', 'wce', 'focal', 'ls'])
    ap.add_argument('--sampler', action='store_true', help='class-balanced WeightedRandomSampler')
    ap.add_argument('--mixup', type=float, default=0.0, help='mixup Beta(alpha, alpha), e.g. 0.2 (0 = off)')
    ap.add_argument('--init', default='', help='run id whose fold{k}.pt initialises fold k (e.g. R8_sam for a 768 fine-tune)')
    ap.add_argument('--hflip', action='store_true', help='random horizontal flip in training + flip TTA at prediction')
    ap.add_argument('--norm', action='store_true', help='per-image 0.5-99.5 percentile intensity window (all splits)')
    ap.add_argument('--sam', type=float, default=0.0, help='SAM neighbourhood radius rho, e.g. 0.05 (0 = off; ~2x step cost)')
    ap.add_argument('--ema', type=float, default=0.0, help='EMA decay per step, e.g. 0.998 (0 = off)')
    ap.add_argument('--only', default='', help='specialist, e.g. bacteria,virus: train on these classes only; '
                                                  'OOF metrics are then not 4-class comparable and no registry row is written')
    ap.add_argument('--note', default='')
    args = ap.parse_args()
    assert args.size <= max(CACHES) and (cache_for(args.size) / 'manifest.json').exists(), \
        f'no finished cache for size {args.size}; run prepare_input.py {max(CACHES)}'

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    out = ROOT / 'outputs' / args.run
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    folds = pd.read_csv(ROOT / 'data/folds.csv')

    t0, parts, fold_f1, best_eps, history = time.time(), [], [], [], []
    for k in map(int, args.folds.split(',')):
        va, probs, f1, ep, hist = run_fold(k, folds, args, out)
        parts.append(va.assign(**{f'p_{c}': probs[:, i] for i, c in enumerate(CLASSES)}))
        fold_f1.append(f1); best_eps.append(ep); history += hist
    pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
    oof = pd.concat(parts)
    oof.to_csv(out / 'oof.csv', index=False)

    y, p = oof.label.values, oof[[f'p_{c}' for c in CLASSES]].values.argmax(1)
    pr, rc, f1s, sup = precision_recall_fscore_support(y, p, labels=[0, 1, 2, 3], zero_division=0)
    metrics = dict(run=args.run, oof_macro_f1=macro_f1(y, p), oof_accuracy=float((y == p).mean()),
                   fold_macro_f1=fold_f1, fold_mean=float(np.mean(fold_f1)), fold_std=float(np.std(fold_f1)),
                   best_epochs=best_eps, minutes=(time.time() - t0) / 60,
                   per_class={c: dict(precision=pr[i], recall=rc[i], f1=f1s[i], support=int(sup[i]))
                              for i, c in enumerate(CLASSES)},
                   confusion_matrix_rows_true=confusion_matrix(y, p, labels=[0, 1, 2, 3]).tolist())
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2, default=float), encoding='utf-8')
    print(json.dumps(metrics, indent=1, default=float))

    if not args.limit and not args.only:
        reg = ROOT / 'experiments.csv'
        row = dict(run_id=args.run, date=datetime.now().strftime('%Y-%m-%d %H:%M'), model=args.model, size=args.size,
                   loss=args.loss + ('+sampler' if args.sampler else ''),
                   augment=args.aug + ('+clahe' if args.clahe else ''), other=args.note, oof_macro_f1=round(metrics['oof_macro_f1'], 4),
                   fold_mean_std=f"{metrics['fold_mean']:.4f}±{metrics['fold_std']:.4f}",
                   **{f'f1_{c}': round(f1s[i], 4) for i, c in enumerate(CLASSES)}, kaggle='', minutes=round(metrics['minutes'], 1))
        new = not reg.exists()
        with reg.open('a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)


if __name__ == '__main__':
    main()
