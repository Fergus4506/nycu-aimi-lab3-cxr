"""Grad-CAM on out-of-fold images: each image is explained by the fold model that did NOT train on it.

Example:  .venv\\Scripts\\python.exe gradcam.py --run A2
Writes outputs/analysis/gradcam_<run>.png (rows: correct bacteria, correct virus, bacteria->virus,
virus->bacteria, COVID-19). Supports ResNet (layer4) and ConvNeXt (features[-1]) checkpoints.
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from train import CACHE, CLASSES, MEAN, STD, build_model, letterbox

ROOT = Path(__file__).resolve().parent
P = [f'p_{c}' for c in CLASSES]


def target_layer(model, name):
    if name.startswith('resnet'):
        return model.layer4
    if name.startswith('convnext') or name.startswith('efficientnet') or name.startswith('densenet'):
        return model.features[-1]
    raise ValueError(name)


def cam(model, layer, x, cls):
    acts, grads = {}, {}
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__('a', o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__('g', go[0]))
    try:
        model.zero_grad()
        out = model(x)
        out[0, cls].backward()
    finally:
        h1.remove(); h2.remove()
    a, g = acts['a'][0], grads['g'][0]                      # C,H,W
    w = g.mean((1, 2))
    c = torch.relu((w[:, None, None] * a).sum(0))
    c = (c / (c.max() + 1e-8)).detach().float().cpu().numpy()
    return cv2.resize(c, (x.shape[3], x.shape[2]))


def pick(oof, rng_seed=0):
    oof = oof.assign(pred=oof[P].values.argmax(1))
    rows = [('correct bacteria', (oof.label == 1) & (oof.pred == 1)),
            ('correct virus', (oof.label == 2) & (oof.pred == 2)),
            ('bacteria -> virus', (oof.label == 1) & (oof.pred == 2)),
            ('virus -> bacteria', (oof.label == 2) & (oof.pred == 1)),
            ('COVID-19 (true)', oof.label == 3)]
    return [(t, oof[m].sample(5, random_state=rng_seed)) for t, m in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='A2')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    out = ROOT / 'outputs' / args.run
    oof = pd.read_csv(out / 'oof.csv')
    picks = pick(oof, args.seed)
    models = {}
    fig, ax = plt.subplots(len(picks), 5, figsize=(16, 3.4 * len(picks)))
    for r, (title, sel) in enumerate(picks):
        for j, (_, x) in enumerate(sel.iterrows()):
            k = int(x.fold)
            if k not in models:
                ck = torch.load(out / f'fold{k}.pt', map_location='cpu', weights_only=False)
                assert ck['classes'] == CLASSES and ck['fold'] == k
                m = build_model(ck['args']['model'], ck['args']['size']); m.load_state_dict(ck['state_dict'])
                models[k] = (m.cuda().eval(), ck['args'])
            model, a = models[k]
            img = letterbox(np.load(CACHE / f'{x.split}_images' / f'{x.new_filename}.npy'), a['size'])
            t = torch.from_numpy(((img.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())[None].cuda()
            pred = int(np.argmax(x[P].values))
            c = cam(model, target_layer(model, a['model']), t.requires_grad_(), pred)
            ax[r, j].imshow(img)
            ax[r, j].imshow(c, cmap='jet', alpha=0.35)
            ax[r, j].axis('off')
            ax[r, j].set_title(f'{title}\ncam for {CLASSES[pred]} p={x[P].values.max():.2f}', fontsize=8)
    fig.suptitle(f'Grad-CAM ({args.run}, out-of-fold models, map for the predicted class)')
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    dst = ROOT / 'outputs/analysis' / f'gradcam_{args.run}.png'
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=100); plt.close(fig)
    print(dst.relative_to(ROOT))


if __name__ == '__main__':
    main()
