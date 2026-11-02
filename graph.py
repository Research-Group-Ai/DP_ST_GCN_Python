import numpy as np
import torch
from torch import nn
# Kinect v2 25-joint topology, zero indexed, child -> parent
PARENTS = [0,0,1,2,20,4,5,6,20,8,9,10,0,12,13,14,0,16,17,18,2,7,7,11,11]
CROSS = [0,1,2,7,11,5,9,6,10]
def adjacency(dual=True, cross=True):
    n=50 if dual else 25
    a=np.eye(n,dtype=np.float32)
    for offset in ([0,25] if dual else [0]):
        for child,parent in enumerate(PARENTS):
            a[offset+child,offset+parent]=a[offset+parent,offset+child]=1
    if dual and cross:
        for j in CROSS: a[j,j+25]=a[j+25,j]=1
    d=np.sum(a,axis=1)
    return torch.tensor(a/np.sqrt(d[:,None]*d[None,:]),dtype=torch.float32)
