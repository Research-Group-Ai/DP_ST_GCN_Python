import argparse,json,random
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import SkeletonDataset
from model import DPSTGCN,IndependentBaseline

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def run(args):
    seed_all(args.seed); device='cuda' if torch.cuda.is_available() else 'cpu'
    train=SkeletonDataset(args.manifest,'train',args.stream,augment=True)
    val=SkeletonDataset(args.manifest,'val',args.stream)
    tr=DataLoader(train,batch_size=args.batch,shuffle=True,num_workers=args.workers)
    va=DataLoader(val,batch_size=args.batch,num_workers=args.workers)
    model=(IndependentBaseline() if args.baseline else DPSTGCN(cross=not args.no_cross)).to(device)
    counts=np.bincount([int(r['label']) for r in train.rows],minlength=51)
    weights=1/np.sqrt(np.maximum(counts,1)); weights=weights/weights.mean()
    loss_fn=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device),label_smoothing=.1)
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0005)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
    scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda')
    best=-1;wait=0;history=[]
    for epoch in range(args.epochs):
        stats=[]
        for training,loader in [(True,tr),(False,va)]:
            model.train(training); total=correct=0; losses=0
            for x,y in loader:
                x,y=x.to(device),y.to(device)
                with torch.set_grad_enabled(training):
                    with torch.autocast(device_type=device,enabled=device=='cuda'):
                        logits=model(x);loss=loss_fn(logits,y)
                    if training:
                        opt.zero_grad();scaler.scale(loss).backward();scaler.step(opt);scaler.update()
                total+=len(y);correct+=(logits.argmax(1)==y).sum().item();losses+=loss.item()*len(y)
            stats.append({'loss':losses/total,'accuracy':correct/total})
        sched.step();history.append({'epoch':epoch+1,'train':stats[0],'val':stats[1]})
        print(f"Epoch {epoch+1}: train={stats[0]['accuracy']:.4f} val={stats[1]['accuracy']:.4f}")
        if stats[1]['accuracy']>best:
            best=stats[1]['accuracy'];wait=0
            torch.save({'model':model.state_dict(),'stream':args.stream,'baseline':args.baseline,'cross':not args.no_cross},args.output)
        else:
            wait+=1
            if wait>=12:break
    with open(args.output+'.history.json','w') as f:json.dump(history,f,indent=2)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--stream',default='joint',choices=['joint','bone','joint_motion','bone_motion']);p.add_argument('--output',default='model.pt');p.add_argument('--epochs',type=int,default=80);p.add_argument('--batch',type=int,default=64);p.add_argument('--workers',type=int,default=0);p.add_argument('--seed',type=int,default=42);p.add_argument('--baseline',action='store_true');p.add_argument('--no-cross',action='store_true');run(p.parse_args())
