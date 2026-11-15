"""EDA, duplicate grouping and 5-fold split for Lab 3 (plan sections 5-6).

Reads data/raw (labels, original headers) and data/cache/long512_npy (pixels).
Writes outputs/eda/* and data/folds.csv. No training.
"""
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent
RAW, CACHE = ROOT / 'data/raw', ROOT / 'data/cache/long512_npy'
OUT = ROOT / 'outputs/eda'
CLASSES = ['normal', 'bacteria', 'virus', 'COVID-19']
SEED, N_FOLDS = 42, 5
HASH = 16           # dHash grid -> 256-bit hash
NEAR_DUP = 8        # max Hamming distance (of 256); visual check: d<=5 same image, d=11 (2833 vs 1923) different patients


def load_index():
    rows = []
    for split in ('train', 'val'):
        d = pd.read_csv(RAW / f'{split}_data.csv')
        d['label'] = d[CLASSES].to_numpy().argmax(1)
        d['split'] = split
        rows.append(d[['new_filename', 'split', 'label']])
    t = pd.read_csv(ROOT / 'test_data_sample.csv')[['new_filename']]
    t['split'], t['label'] = 'test', -1
    df = pd.concat(rows + [t], ignore_index=True)
    assert len(df) == 5908 and df.new_filename.is_unique
    return df


def image_stats(df):
    recs = []
    for name, split in zip(df.new_filename, df.split):
        with Image.open(RAW / f'{split}_images' / name) as im:
            w, h, mode, fmt = im.width, im.height, im.mode, im.format
        a = np.load(CACHE / f'{split}_images' / f'{name}.npy')
        rgb_equal = bool((a[..., 0] == a[..., 1]).all() and (a[..., 1] == a[..., 2]).all())
        g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(g, (HASH + 1, HASH), interpolation=cv2.INTER_AREA)
        dhash = (small[:, 1:] > small[:, :-1]).flatten()
        b = max(1, min(g.shape) // 50)  # outer 2% frame
        border = np.concatenate([g[:b].ravel(), g[-b:].ravel(), g[:, :b].ravel(), g[:, -b:].ravel()])
        recs.append(dict(width=w, height=h, mode=mode, format=fmt, ext=name.rsplit('.', 1)[-1].lower(),
                         rgb_equal=rgb_equal, mean=g.mean(), std=g.std(), border_mean=border.mean(),
                         dark_frac=(g < 10).mean(), file_kb=(RAW / f'{split}_images' / name).stat().st_size / 1024,
                         dhash=dhash))
    s = pd.DataFrame(recs)
    s['aspect'] = s.width / s.height
    s['number'] = pd.to_numeric(df.new_filename.str.split('.').str[0], errors='coerce')
    return pd.concat([df, s], axis=1)


def duplicate_groups(df):
    h = np.stack(df.dhash.to_numpy()).astype(np.float32)
    dist = h @ (1 - h).T + (1 - h) @ h.T          # pairwise Hamming distance
    np.fill_diagonal(dist, 1e9)
    nn = dist.min(1)
    parent = list(range(len(df)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    ii, jj = np.where(np.triu(dist <= NEAR_DUP, 1))
    for i, j in zip(ii, jj):
        parent[find(i)] = find(j)
    pairs = pd.DataFrame({'a': df.new_filename.values[ii], 'a_split': df.split.values[ii], 'a_label': df.label.values[ii],
                          'b': df.new_filename.values[jj], 'b_split': df.split.values[jj], 'b_label': df.label.values[jj],
                          'distance': dist[ii, jj].astype(int)}).sort_values('distance')
    return np.array([find(i) for i in range(len(df))]), nn, pairs


def make_folds(lab):
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = np.full(len(lab), -1)
    for k, (_, va) in enumerate(sgkf.split(lab, lab.label, lab.group)):
        fold[va] = k
    lab = lab.assign(fold=fold)
    assert (lab.fold >= 0).all()
    assert lab.groupby('group').fold.nunique().max() == 1, 'a duplicate group spans folds'
    support = pd.crosstab(lab.fold, lab.label).reindex(columns=range(4), fill_value=0)
    assert (support > 0).all().all(), f'a fold lacks a class:\n{support}'
    return lab, support


def load_gray(row, size=256):
    a = np.load(CACHE / f'{row.split}_images' / f'{row.new_filename}.npy')
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    s = size / max(g.shape)
    g = cv2.resize(g, (round(g.shape[1] * s), round(g.shape[0] * s)), interpolation=cv2.INTER_AREA)
    c = np.zeros((size, size), np.uint8)
    c[(size - g.shape[0]) // 2:(size - g.shape[0]) // 2 + g.shape[0], (size - g.shape[1]) // 2:(size - g.shape[1]) // 2 + g.shape[1]] = g
    return c


def figures(df, lab, nn, pairs, support):
    labeled = df[df.label >= 0]
    names = [CLASSES[i] for i in labeled.label]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    pd.crosstab(labeled.label.map(dict(enumerate(CLASSES))), labeled.split).loc[CLASSES].plot.bar(ax=ax, rot=0)
    ax.set_ylabel('images'); ax.set_title('Class distribution (train / val)')
    fig.tight_layout(); fig.savefig(OUT / 'class_distribution.png', dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for c, colr in zip(range(4), ['C0', 'C1', 'C2', 'C3']):
        d = labeled[labeled.label == c]
        axes[0].scatter(d.width, d.height, s=4 if c < 3 else 18, alpha=.4 if c < 3 else 1, c=colr, label=CLASSES[c])
        axes[1].hist(d['mean'], bins=40, density=True, histtype='step', color=colr, label=CLASSES[c])
        axes[2].hist(d.aspect, bins=40, density=True, histtype='step', color=colr, label=CLASSES[c])
    axes[0].set(xlabel='original width (px)', ylabel='original height (px)', xscale='log', yscale='log', title='Original size by class')
    axes[1].set(xlabel='mean intensity (0-255)', ylabel='density', title='Brightness by class')
    axes[2].set(xlabel='width / height', ylabel='density', title='Aspect ratio by class')
    axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for a in axes: a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / 'size_brightness_by_class.png', dpi=150); plt.close(fig)

    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(4, 8, figsize=(16, 8.4))
    for c in range(4):
        d = labeled[labeled.label == c]
        for k, idx in enumerate(rng.choice(d.index, 8, replace=False)):
            axes[c, k].imshow(load_gray(df.loc[idx]), cmap='gray'); axes[c, k].axis('off')
            axes[c, k].set_title(f'{CLASSES[c]} {df.new_filename[idx]}', fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / 'samples_by_class.png', dpi=120); plt.close(fig)

    covid = labeled[labeled.label == 3]
    n = len(covid); cols = 10; rows_ = -(-n // cols)
    fig, axes = plt.subplots(rows_, cols, figsize=(cols * 1.8, rows_ * 1.95))
    for k, ax in enumerate(axes.flat):
        ax.axis('off')
        if k < n:
            r = covid.iloc[k]
            ax.imshow(load_gray(r), cmap='gray'); ax.set_title(f'{r.new_filename} {r.width}x{r.height}', fontsize=6)
    fig.tight_layout(); fig.savefig(OUT / 'covid_all.png', dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(nn, bins=np.arange(0, 120, 2)); ax.axvline(NEAR_DUP + .5, color='r', ls='--', label=f'threshold {NEAR_DUP}')
    ax.set(xlabel='Hamming distance to nearest other image (256-bit dHash)', ylabel='images', title='Near-duplicate distance')
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / 'dhash_nearest_distance.png', dpi=150); plt.close(fig)

    show = pd.concat([pairs[pairs.distance > 0], pairs[pairs.distance == 0].head(4)])  # all non-exact pairs need eyes; exact ones are pixel-identical
    if len(show):
        fig, axes = plt.subplots(len(show), 2, figsize=(4, 2 * len(show)))
        axes = np.atleast_2d(axes)
        for r, p in enumerate(show.itertuples()):
            for c, (nm, sp) in enumerate([(p.a, p.a_split), (p.b, p.b_split)]):
                axes[r, c].imshow(load_gray(pd.Series({'new_filename': nm, 'split': sp})), cmap='gray')
                axes[r, c].axis('off'); axes[r, c].set_title(f'{sp}/{nm} d={p.distance}', fontsize=6)
        fig.tight_layout(); fig.savefig(OUT / 'near_duplicate_pairs.png', dpi=100); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    (support / support.sum()).rename(columns=dict(enumerate(CLASSES))).plot.bar(ax=ax, rot=0)
    ax.set(xlabel='fold', ylabel='class fraction', title='Class fraction per fold'); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / 'fold_class_fraction.png', dpi=150); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = image_stats(load_index())
    group, nn, pairs = duplicate_groups(df)
    df['group'] = group
    df['nn_distance'] = nn.astype(int)

    lab = df[df.label >= 0].copy()
    lab, support = make_folds(lab)
    lab[['new_filename', 'split', 'label', 'group', 'fold']].to_csv(ROOT / 'data/folds.csv', index=False)

    df.drop(columns='dhash').to_csv(OUT / 'image_stats.csv', index=False)
    pairs.to_csv(OUT / 'near_duplicate_pairs.csv', index=False)
    figures(df, lab, nn, pairs, support)

    labeled = df[df.label >= 0]
    by_class = labeled.groupby('label').agg(
        n=('new_filename', 'size'), width_median=('width', 'median'), height_median=('height', 'median'),
        width_min=('width', 'min'), width_max=('width', 'max'), aspect_median=('aspect', 'median'),
        mean_intensity=('mean', 'mean'), std_intensity=('std', 'mean'), border_mean=('border_mean', 'mean'),
        dark_frac=('dark_frac', 'mean'), file_kb_median=('file_kb', 'median'),
        number_min=('number', 'min'), number_max=('number', 'max'), rgb_equal_frac=('rgb_equal', 'mean'))
    by_class.index = CLASSES
    dup_sizes = pd.Series(group).value_counts()
    multi = df[df.group.isin(dup_sizes[dup_sizes > 1].index)]
    cross = multi.groupby('group').agg(splits=('split', lambda s: ','.join(sorted(set(s)))),
                                       labels=('label', lambda s: sorted(set(x for x in s if x >= 0))))
    summary = {
        'counts': pd.crosstab(labeled.split, labeled.label.map(dict(enumerate(CLASSES)))).to_dict(),
        'by_class': json.loads(by_class.round(3).to_json(orient='index')),
        'ext_by_class': pd.crosstab(labeled.ext, labeled.label.map(dict(enumerate(CLASSES)))).to_dict(),
        'test_ext': df[df.split == 'test'].ext.value_counts().to_dict(),
        'mode_by_class': pd.crosstab(labeled['mode'], labeled.label.map(dict(enumerate(CLASSES)))).to_dict(),
        'format_by_class': pd.crosstab(labeled['format'], labeled.label.map(dict(enumerate(CLASSES)))).to_dict(),
        'near_dup': {'threshold': NEAR_DUP, 'hash_bits': HASH * HASH, 'pairs': len(pairs),
                     'groups_gt1': int((dup_sizes > 1).sum()), 'images_in_groups': int(len(multi)),
                     'groups_cross_split': int((cross.splits.str.contains(',')).sum()),
                     'groups_label_conflict': int((cross.labels.map(len) > 1).sum()),
                     'groups_touching_test': int(cross.splits.str.contains('test').sum())},
        'folds': {'method': f'StratifiedGroupKFold(n_splits={N_FOLDS}, shuffle=True, random_state={SEED}) on train+val, groups = near-duplicate clusters',
                  'support': {int(k): {CLASSES[c]: int(v) for c, v in row.items()} for k, row in support.iterrows()}},
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
