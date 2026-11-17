"""Measure batch-1 model latency for one or two streams; excludes preprocessing."""
import argparse
import json
import time
import numpy as np
import torch
from dpstgcn.model import Recognizer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--bone-checkpoint')
    p.add_argument('--joint-weight',type=float,default=.5)
    p.add_argument('--iterations',type=int,default=100)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if a.iterations<2 or not 0<=a.joint_weight<=1: raise ValueError('Invalid iterations or weight.')
    device=torch.device(a.device); models=[]; frames=[]
    for path in [a.checkpoint]+([a.bone_checkpoint] if a.bone_checkpoint else []):
        ck=torch.load(path,map_location='cpu',weights_only=False)
        m=Recognizer(ck['config'],ck['mode']).to(device); m.load_state_dict(ck['model']); m.eval()
        models.append(m); frames.append(ck['config']['frames'])
    inputs=[torch.randn(1,3,t,50,device=device) for t in frames]
    def sync():
        if device.type=='cuda': torch.cuda.synchronize()
    def predict():
        probabilities=[m(x)[0].softmax(1) for m,x in zip(models,inputs)]
        return probabilities[0] if len(models)==1 else a.joint_weight*probabilities[0]+(1-a.joint_weight)*probabilities[1]
    times=[]
    with torch.inference_mode():
        for _ in range(10): predict()
        sync()
        for _ in range(a.iterations):
            sync(); start=time.perf_counter(); predict(); sync()
            times.append((time.perf_counter()-start)*1000)
    result=dict(device=str(device),torch_version=torch.__version__,batch_size=1,
       streams=len(models),median_ms=float(np.median(times)),mean_ms=float(np.mean(times)),
       p95_ms=float(np.percentile(times,95)),parameters=sum(p.numel() for m in models for p in m.parameters()),
       scope='Model-only synthetic inputs; excludes loading, preprocessing, transfers and skeleton estimation. Two-stream mode directly times both forwards and probability fusion.')
    from pathlib import Path
    output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))


if __name__=='__main__': main()
