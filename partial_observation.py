"""Evaluate observed-prefix masking for one or more trained checkpoints."""
import argparse,csv
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import SkeletonDataset
from evaluate import load,predict
import torch
import numpy as np
from sklearn.metrics import accuracy_score

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--checkpoints',nargs='+',required=True);p.add_argument('--output',default='partial_observation.csv');p.add_argument('--batch',type=int,default=64);p.add_argument('--ratios',nargs='+',type=float,default=[.1,.2,.3,.4,.5,.6,.7,.8,.9,1.]);a=p.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu';rows=[]
    for path in a.checkpoints:
        model,stream=load(path,device);loader=DataLoader(SkeletonDataset(a.manifest,'val',stream),batch_size=a.batch)
        for ratio in a.ratios:
            if not 0<ratio<=1:raise ValueError('ratios must be in (0,1]')
            probs,labels,_=predict(model,loader,device,ratio)
            rows.append({'stream':stream,'checkpoint':Path(path).name,'ratio':ratio,'accuracy':accuracy_score(labels,probs.argmax(1))})
    with open(a.output,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print('Saved',a.output)
