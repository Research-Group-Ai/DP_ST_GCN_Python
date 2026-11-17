"""Export prediction probabilities, per-class metrics, attention and timing."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from dpstgcn.data import Clips
from dpstgcn.model import Recognizer
from dpstgcn.metrics import save_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--split',choices=['train','val','test'],default='val')
    p.add_argument('--fraction',type=float,default=1.)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args()
    # Only load locally created/trusted checkpoint files.
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    c=ck['config']; device=torch.device(a.device)
    model=Recognizer(c,ck['mode']).to(device)
    model.load_state_dict(ck['model']); model.eval()
    data=Clips(a.manifest,a.split,ck['stream'],False,c['frames'],a.fraction)
    loader=DataLoader(data,batch_size=a.batch_size,shuffle=False)
    probabilities=[]; labels=[]; ids=[]; attention=[]
    elapsed=0.
    with torch.inference_mode():
        # Warm-up is excluded. Timing excludes loading, preprocessing and transfer.
        example=next(iter(loader))[0].to(device)
        for _ in range(5): model(example)
        if device.type=='cuda': torch.cuda.synchronize()
        for x,y,clip_ids in loader:
            x=x.to(device)
            if device.type=='cuda': torch.cuda.synchronize()
            start=time.perf_counter()
            logits,alpha=model(x)
            if device.type=='cuda': torch.cuda.synchronize()
            elapsed+=time.perf_counter()-start
            probabilities.append(logits.softmax(1).cpu().numpy())
            attention.append(alpha.cpu().numpy()); labels.extend(y.tolist()); ids.extend(clip_ids)
    a.output.mkdir(parents=True,exist_ok=True)
    probs=np.concatenate(probabilities)
    np.savez_compressed(a.output/'predictions.npz',probabilities=probs,labels=np.array(labels),
                       ids=np.array(ids),attention=np.concatenate(attention),split=a.split,
                       fraction=a.fraction,stream=ck['stream'],mode=ck['mode'])
    save_metrics(labels,probs,a.output,c['classes'])
    (a.output/'timing.json').write_text(json.dumps(dict(device=str(device),
       batch_size=a.batch_size,model_only_ms_per_clip=1000*elapsed/len(data),
       parameters=sum(p.numel() for p in model.parameters()),
       caveat='Batched amortized throughput timing; excludes preprocessing, transfer and skeleton estimation. Not batch-1 latency.'),indent=2))


if __name__=='__main__': main()
