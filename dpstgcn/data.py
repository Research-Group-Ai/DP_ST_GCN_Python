import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .preprocess import prepare_clip,representation


def load_manifest(path):
    path=Path(path)
    rows=json.loads(path.read_text())
    ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)):
        raise ValueError('Duplicate clip IDs.')
    for row in rows:
        if row['split'] not in ('train','val','test'):
            raise ValueError('split must be train, val or test.')
        if not isinstance(row['label'],int) or not 0<=row['label']<51:
            raise ValueError('Labels must be zero-based integers [0,50].')
        if not row.get('subjects'):
            raise ValueError('Subject metadata required to audit cross-subject separation.')
    for key in ('subjects','recording'):
        sets={s:set() for s in ('train','val','test')}
        for row in rows:
            values=row[key] if key=='subjects' else [row[key]]
            sets[row['split']].update(str(v) for v in values)
        for a,b in (('train','val'),('train','test'),('val','test')):
            if sets[a]&sets[b]:
                raise ValueError(f'{key} leakage between {a} and {b}.')
    return rows


class Clips(Dataset):
    def __init__(self,manifest,split,stream='joint',training=False,frames=100,fraction=1.):
        self.path=Path(manifest)
        self.rows=[r for r in load_manifest(manifest) if r['split']==split]
        if not self.rows:
            raise ValueError(f'No {split} records.')
        self.stream,self.training,self.frames,self.fraction=stream,training,frames,fraction

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,i):
        row=self.rows[i]
        raw=np.load(self.path.parent/row['path'],allow_pickle=False)
        # numpy global RNG is seeded per worker or by training seed.
        rng=np.random.default_rng(np.random.randint(0,2**32-1))
        x,mask=prepare_clip(raw,self.frames,self.training,rng,self.fraction)
        return torch.from_numpy(representation(x,mask,self.stream)),row['label'],row['id']


def seed_worker(worker_id):
    seed=torch.initial_seed()%2**32
    np.random.seed(seed)
    random.seed(seed)
