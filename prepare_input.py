"""Build a verified resize cache and benchmark Windows DataLoader input only."""
import csv
import ctypes
import hashlib
import json
import random
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data/raw'
CACHE = ROOT / 'data/cache/long512_npy'
OUT = ROOT / 'outputs/preflight'


def resized(path, long_edge=512):
    with Image.open(path) as im:
        array = np.array(im)
    if array.dtype not in (np.uint8, np.uint16) or array.ndim not in (2, 3):
        raise ValueError(f'Unsupported pixel representation: {path}')
    h, w = array.shape[:2]
    scale = min(1.0, long_edge / max(h, w))
    if scale < 1:
        array = cv2.resize(array, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return array


def build_cache(long_edge=512):
    cache = ROOT / f'data/cache/long{long_edge}_npy'
    start = time.perf_counter()
    manifest = []
    for split in ('train', 'val', 'test'):
        for path in sorted((RAW / f'{split}_images').rglob('*')):
            if path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                continue
            relative = path.relative_to(RAW)
            target = cache / relative.with_suffix(relative.suffix + '.npy')
            target.parent.mkdir(parents=True, exist_ok=True)
            array = resized(path, long_edge)
            np.save(target, array, allow_pickle=False)
            loaded = np.load(target, allow_pickle=False)
            assert loaded.dtype == array.dtype and np.array_equal(loaded, array)
            manifest.append({'source':relative.as_posix(), 'cache':target.relative_to(cache).as_posix(),
                'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'cache_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                'shape':list(array.shape), 'dtype':str(array.dtype), 'bytes':target.stat().st_size})
    assert len(manifest) == 5908
    result = {'long_edge':long_edge, 'upsample':False, 'interpolation':'INTER_AREA',
        'pixels':'Original channel count and uint8/uint16 dtype preserved; resizing is lossy.',
        'count':len(manifest), 'seconds':time.perf_counter()-start,
        'bytes':sum(x['bytes'] for x in manifest), 'files':manifest}
    (cache / 'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='files'}), flush=True)
    return result


class Inputs(Dataset):
    def __init__(self, names, cached):
        self.names, self.cached = names, cached

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        array = (np.load(CACHE / 'train_images' / (name+'.npy'), allow_pickle=False)
                 if self.cached else resized(RAW / 'train_images' / name))
        maximum = np.iinfo(array.dtype).max
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        if array.shape[2] != 3:
            raise ValueError('Expected grayscale or RGB input')
        h,w = array.shape[:2]
        scale = 384 / max(h,w)
        image = cv2.resize(array, (max(1,round(w*scale)),max(1,round(h*scale))), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((384,384,3), dtype=array.dtype)
        hh,ww=image.shape[:2]
        canvas[(384-hh)//2:(384-hh)//2+hh,(384-ww)//2:(384-ww)//2+ww] = image
        return torch.from_numpy(canvas.transpose(2,0,1).copy()).float().div_(maximum), index


def worker_init(_):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


class MemoryStatus(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in
        ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]


def used_ram():
    status=MemoryStatus(); status.length=ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return (status.total_phys-status.avail_phys)/2**30


def benchmark():
    buckets=[[] for _ in range(4)]
    with (RAW/'train_data.csv').open(newline='',encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            label=[row[c] for c in ('normal','bacteria','virus','COVID-19')].index('1')
            buckets[label].append(row['new_filename'])
    rng=random.Random(42)
    names=[]
    for bucket in buckets:
        names.extend(rng.sample(sorted(bucket),min(128,len(bucket))))
    rng.shuffle(names)
    assert len(names)==423 and len(set(names))==423
    # Equivalent preprocessing makes this an I/O comparison, not an accuracy experiment.
    raw,cached=Inputs(names,False),Inputs(names,True)
    for i in range(len(names)):
        assert torch.equal(raw[i][0],cached[i][0])
    results=[]
    for cached_mode in (False,True):
        for workers in (0,2,4,8):
            samples=[used_ram()]; stop=threading.Event()
            def monitor():
                while not stop.wait(.1): samples.append(used_ram())
            monitor_thread=threading.Thread(target=monitor,daemon=True); monitor_thread.start()
            loader=DataLoader(Inputs(names,cached_mode),batch_size=32,shuffle=False,num_workers=workers,
                pin_memory=True,persistent_workers=workers>0,worker_init_fn=worker_init,
                **({'prefetch_factor':2,'multiprocessing_context':'spawn'} if workers else {}))
            timings=[]
            try:
                for epoch in range(2):
                    seen=[]; torch.cuda.synchronize(); start=time.perf_counter()
                    for batch,indices in loader:
                        assert batch.shape[1:]==(3,384,384) and torch.isfinite(batch).all()
                        batch.to('cuda',non_blocking=True)
                        seen.extend(indices.tolist())
                    torch.cuda.synchronize(); timings.append(time.perf_counter()-start)
                    assert seen==list(range(len(names)))
                row={'cached':cached_mode,'workers':workers,'samples':len(names),
                    'first_epoch_s':timings[0],'steady_epoch_s':timings[1],
                    'steady_images_s':len(names)/timings[1],
                    'system_ram_start_gib':samples[0],'system_ram_peak_gib':max(samples)}
                results.append(row); print(json.dumps(row),flush=True)
                OUT.mkdir(parents=True,exist_ok=True)
                (OUT/'loader_benchmark.json').write_text(json.dumps({'seed':42,'batch_size':32,'input_size':384,
                    'sample_names':names,'method':'423 stratified train samples; raw decode versus NPY; identical resize/padding; no augmentation/model; pinned CPU to CUDA; persistent workers; epoch 1 includes startup; epoch 2 steady. RAM is whole-system, not process-specific. OS cache not flushed.',
                    'results':results},indent=2),encoding='utf-8')
            finally:
                del loader
                stop.set(); monitor_thread.join()
    return results


if __name__=='__main__':
    import sys
    if len(sys.argv) > 1:  # e.g. `prepare_input.py 768`: build only that cache, no benchmark
        build_cache(int(sys.argv[1]))
    else:
        worker_init(0)
        assert torch.cuda.is_available()
        build_cache()
        benchmark()
