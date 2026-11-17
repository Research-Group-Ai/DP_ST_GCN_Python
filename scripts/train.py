import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from dpstgcn.data import Clips,seed_worker
from dpstgcn.model import Recognizer


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--config',default='configs/model.json',type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--stream',choices=['joint','bone','joint_motion','bone_motion'],default='joint')
    p.add_argument('--mode',choices=['dual','single','dual_no_cross'],default='dual')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args()
    c=json.loads(a.config.read_text())
    if not c['protocol_verified'] and not c.get('synthetic_demo'):
        raise ValueError('Verify graph, normalization, frame indexing, release and subject split; set protocol_verified=true. See docs.')
    seed=c['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=torch.device(a.device)
    train=Clips(a.manifest,'train',a.stream,True,c['frames'])
    val=Clips(a.manifest,'val',a.stream,False,c['frames'])
    counts=np.bincount([r['label'] for r in train.rows],minlength=c['classes'])
    if (counts==0).any():
        raise ValueError('Every class needs training examples for inverse-frequency weights.')
    weights=1/np.sqrt(counts.astype(float))
    weights/=weights.mean()  # explicit normalization convention, not specified in paper
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device),label_smoothing=c['label_smoothing'])
    generator=torch.Generator().manual_seed(seed)
    loaders={s:DataLoader(d,batch_size=c['batch_size'],shuffle=s=='train',
                  num_workers=c['workers'],worker_init_fn=seed_worker,generator=generator,
                  pin_memory=device.type=='cuda') for s,d in [('train',train),('val',val)]}
    model=Recognizer(c,a.mode).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=c['epochs'])
    amp=device.type=='cuda'
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    a.output.mkdir(parents=True,exist_ok=True)
    best=-1.; stale=0; history=[]
    for epoch in range(c['epochs']):
        record={'epoch':epoch+1,'learning_rate':optimizer.param_groups[0]['lr']}
        for split,loader in loaders.items():
            model.train(split=='train')
            correct=0; total=0; loss_sum=0.
            for x,y,_ in loader:
                x,y=x.to(device),y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(split=='train'):
                    with torch.autocast(device_type=device.type,enabled=amp):
                        logits,_=model(x); loss=loss_fn(logits,y)
                    if split=='train':
                        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                loss_sum+=loss.item()*len(y)
                correct+=(logits.argmax(1)==y).sum().item(); total+=len(y)
            record[split+'_loss_batch_mean']=loss_sum/total
            record[split+'_accuracy']=correct/total
        scheduler.step()
        history.append(record)
        print(json.dumps(record),flush=True)
        if record['val_accuracy']>best:
            best=record['val_accuracy']; stale=0
            torch.save(dict(model=model.state_dict(),config=c,mode=a.mode,stream=a.stream,
               epoch=epoch+1,val_accuracy=best,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict()),a.output/'best.pt')
        else: stale+=1
        (a.output/'history.json').write_text(json.dumps(history,indent=2))
        if stale>=c['patience']: break
    manifest=dict(config=c,mode=a.mode,stream=a.stream,torch_version=torch.__version__,
                  numpy_version=np.__version__,device=str(device),seed=seed,
                  parameters=sum(p.numel() for p in model.parameters()),class_counts=counts.tolist(),
                  manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
                  status='SYNTHETIC SMOKE TEST' if c.get('synthetic_demo') else 'RECONSTRUCTED IMPLEMENTATION; NOT VERIFIED PAPER REPRODUCTION')
    (a.output/'run_manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__': main()
