import numpy as np
from graph import PARENTS

def feature(x,stream):
    # x: T,2,25,3; coordinate normalization should be performed upstream
    x=np.asarray(x,dtype=np.float32)
    if x.ndim != 4 or x.shape[1:] != (2,25,3): raise ValueError('Expected (T,2,25,3)')
    if stream in ('bone','bone_motion'):
        x=x-x[:,:,PARENTS,:]
    if stream in ('joint_motion','bone_motion'):
        x=np.concatenate([np.zeros_like(x[:1]),np.diff(x,axis=0)],axis=0)
    return x.reshape(x.shape[0],50,3)

def resample(x,frames=100):
    indices=np.linspace(0,len(x)-1,frames).round().astype(int)
    return x[indices]

def normalize(x):
    # Subtract first person's root, using a common reference to preserve relative person position
    return x-x[:1,:1,:1,:]
