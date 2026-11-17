"""Select lambda on validation predictions or apply a fixed saved lambda."""
import argparse
import json
from pathlib import Path
import numpy as np
from dpstgcn.metrics import save_metrics


def aligned(joint,bone):
    if len(set(joint['ids'].tolist()))!=len(joint['ids']) or len(set(bone['ids'].tolist()))!=len(bone['ids']):
        raise ValueError('Prediction IDs must be unique.')
    lookup={v:i for i,v in enumerate(bone['ids'])}
    if set(lookup)!=set(joint['ids']): raise ValueError('Different clip IDs.')
    ix=[lookup[v] for v in joint['ids']]
    if not np.array_equal(joint['labels'],bone['labels'][ix]): raise ValueError('Label mismatch.')
    if str(joint['split'])!=str(bone['split']) or float(joint['fraction'])!=float(bone['fraction']):
        raise ValueError('Split/observation fraction mismatch.')
    if str(joint['stream'])!='joint' or str(bone['stream'])!='bone': raise ValueError('Expected joint and bone streams.')
    if str(joint['mode'])!=str(bone['mode']): raise ValueError('Graph-mode mismatch.')
    return joint['probabilities'],bone['probabilities'][ix],joint['labels']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--joint',required=True,type=Path); p.add_argument('--bone',required=True,type=Path)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--select-on-validation',action='store_true')
    group.add_argument('--lambda-file',type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    j=np.load(a.joint,allow_pickle=False); b=np.load(a.bone,allow_pickle=False)
    pj,pb,y=aligned(j,b)
    if a.select_on_validation:
        if str(j['split'])!='val' or float(j['fraction'])!=1.:
            raise ValueError('Tune only on full-observation validation predictions.')
        grid=np.linspace(0,1,101) # explicit new search resolution
        scores=[float(((v*pj+(1-v)*pb).argmax(1)==y).mean()) for v in grid]
        index=int(np.argmax(scores)); lam=float(grid[index])
        selection=dict(joint_weight=lam,grid_step=.01,tie_break='smallest lambda',
                       selected_on='validation',status='Validation-tuned; not independent test accuracy')
    else:
        selection=json.loads(a.lambda_file.read_text()); lam=selection['joint_weight']
        if not 0<=lam<=1: raise ValueError('Invalid lambda.')
    output=lam*pj+(1-lam)*pb
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'fusion_weight.json').write_text(json.dumps(selection,indent=2))
    np.savez_compressed(a.output/'predictions.npz',probabilities=output,labels=y,ids=j['ids'],
                       split=j['split'],fraction=j['fraction'])
    save_metrics(y,output,a.output,pj.shape[1])


if __name__=='__main__': main()
