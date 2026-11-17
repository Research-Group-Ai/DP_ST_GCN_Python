import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    data=np.load(a.predictions,allow_pickle=False)
    if 'attention' not in data: raise ValueError('Use a single-stream evaluation archive containing attention.')
    att=data['attention']
    fig,ax=plt.subplots(figsize=(7,4))
    if att.ndim==3:
        for person in range(2): ax.plot(np.linspace(0,1,att.shape[-1]),att[:,person].mean(0),label=f'Person {person+1}')
        ax.legend()
    else: ax.plot(np.linspace(0,1,att.shape[-1]),att.mean(0))
    ax.set(xlabel='Normalized output-frame position',ylabel='Mean temporal attention',title='All evaluated clips')
    fig.tight_layout(); fig.savefig(a.output/'attention.png',dpi=180); plt.close(fig)
    # Classwise attention arrays support separately chosen interaction-class plots.
    arrays={f'class_{int(k)}':att[data['labels']==k].mean(0) for k in np.unique(data['labels'])}
    np.savez_compressed(a.output/'attention_by_class.npz',**arrays)
    matrix=a.predictions.parent/'confusion_normalized.csv'
    if matrix.exists():
        cm=np.loadtxt(matrix,delimiter=',')
        fig,ax=plt.subplots(figsize=(8,7)); im=ax.imshow(cm,vmin=0,vmax=1,cmap='Blues')
        ax.set(xlabel='Predicted class',ylabel='True class'); fig.colorbar(im,ax=ax)
        fig.tight_layout(); fig.savefig(a.output/'confusion.png',dpi=180); plt.close(fig)


if __name__=='__main__': main()
