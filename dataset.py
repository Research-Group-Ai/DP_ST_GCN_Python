import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from features import feature,normalize,resample

class SkeletonDataset(Dataset):
    """CSV columns: path,label,split,subject. NPZ/NPY arrays must be (T,2,25,3)."""
    def __init__(self,manifest,split='train',stream='joint',frames=100,augment=False):
        self.base=Path(manifest).resolve().parent
        with open(manifest,newline='') as f: self.rows=[r for r in csv.DictReader(f) if r['split']==split]
        self.stream,self.frames,self.augment=stream,frames,augment
        if not self.rows: raise ValueError(f'No samples for split={split}')
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; p=self.base/r['path']; loaded=np.load(p)
        x=loaded['skeleton'] if isinstance(loaded,np.lib.npyio.NpzFile) else loaded
        x=resample(np.asarray(x,dtype=np.float32),self.frames)
        x=normalize(x)
        if self.augment:
            angle=np.deg2rad(np.random.uniform(-25,25)); c,s=np.cos(angle),np.sin(angle)
            rot=np.array([[c,0,s],[0,1,0],[-s,0,c]],dtype=np.float32)
            x=x@rot.T*np.random.uniform(.85,1.15)
            x+=np.random.normal(0,.005,x.shape).astype(np.float32)
            if np.random.rand()<.5:
                width=np.random.randint(1,max(2,self.frames//10)); start=np.random.randint(0,self.frames-width+1)
                x[start:start+width]=0
        x=feature(x,self.stream)
        return torch.from_numpy(x).permute(2,0,1).contiguous(),int(r['label'])
