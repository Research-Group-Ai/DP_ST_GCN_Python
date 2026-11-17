"""Missing-aware normalization, joint augmentation and feature streams."""
import numpy as np
from .graph import PARENTS


def valid_mask(x):
    return np.isfinite(x).all(axis=-1) & np.any(x!=0,axis=-1)


def prepare_clip(raw, frames=100, training=False, rng=None, fraction=1.):
    x=np.asarray(raw,dtype=np.float32).copy()
    if x.ndim!=4 or x.shape[1:]!=(2,25,3) or not len(x):
        raise ValueError('Expected nonempty (T,2,25,3) clip.')
    if not 0<fraction<=1:
        raise ValueError('Observation fraction must lie in (0,1].')
    # Truncate BEFORE normalization or resampling: no future-frame information.
    x=x[:max(1,int(np.ceil(len(x)*fraction)))]
    mask=valid_mask(x)
    if not mask[:,0,0].all():
        raise ValueError('Person A spine-base missing; explicit imputation policy required.')
    usable=mask[:,0,0]&mask[:,0,2]
    if not usable.any():
        raise ValueError('No valid person A spine-base/neck pairs.')
    # Explicit interpretation of paper: distance between temporal mean neck/root.
    scale=np.linalg.norm(x[usable,0,2].mean(axis=0)-x[usable,0,0].mean(axis=0))
    if not np.isfinite(scale) or scale<1e-6:
        raise ValueError('Degenerate normalization scale.')
    root=x[:,0,0].copy()
    x=(x-root[:,None,None,:])/scale
    x[~mask]=0
    idx=np.linspace(0,len(x)-1,frames).round().astype(int)
    x,mask=x[idx],mask[idx]
    if training:
        rng=rng if rng is not None else np.random.default_rng()
        angle=np.deg2rad(rng.uniform(-25,25))
        cs,sn=np.cos(angle),np.sin(angle)
        rot=np.array([[cs,0,sn],[0,1,0],[-sn,0,cs]],np.float32)
        x=x@rot.T*rng.uniform(.85,1.15)
        x+=rng.normal(0,.01,x.shape).astype(np.float32)*mask[...,None]
        x[~mask]=0
        if rng.random()<.5:
            start=int(rng.integers(0,max(1,frames-5+1)))
            x[start:start+5]=0
            mask[start:start+5]=False
    return x.astype(np.float32),mask


def representation(x, mask, stream):
    if stream not in ('joint','bone','joint_motion','bone_motion'):
        raise ValueError('Unknown stream')
    y=x.copy()
    ok=mask.copy()
    if stream.startswith('bone'):
        y=x-x[:,:,PARENTS,:]
        ok=mask & mask[:,:,PARENTS]
        y[~ok]=0
        y[:,:,0]=0
    if stream.endswith('motion'):
        # Explicit forward-difference convention, last frame zero.
        motion=np.zeros_like(y)
        adjacent=ok[:-1]&ok[1:]
        motion[:-1]=(y[1:]-y[:-1])*adjacent[...,None]
        y=motion
    # Network convention: (channels,time,person-major joint nodes).
    return y.reshape(len(y),50,3).transpose(2,0,1).copy()
