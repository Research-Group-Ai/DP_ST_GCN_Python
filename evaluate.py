import argparse,time,json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix
from dataset import SkeletonDataset
from model import DPSTGCN,IndependentBaseline

def load(path,device):
    ck=torch.load(path,map_location=device,weights_only=True)
    m=IndependentBaseline() if ck['baseline'] else DPSTGCN(cross=ck['cross'])
    m.load_state_dict(ck['model']);return m.to(device).eval(),ck['stream']
@torch.no_grad()
def predict(model,loader,device,ratio=1):
    probs=[];targets=[];attns=[]
    for x,y in loader:
        x=x.to(device)
        if ratio<1:x[:,:,max(1,int(x.shape[2]*ratio)):,:]=0
        out=model(x,return_attention=True)
        logits,attn=out
        probs.append(logits.softmax(-1).cpu().numpy());targets.extend(y.numpy())
        if attn is not None:attns.extend(attn.cpu().numpy())
    return np.concatenate(probs),np.asarray(targets),np.asarray(attns)
def main(args):
    device='cuda' if torch.cuda.is_available() else 'cpu';m,stream=load(args.checkpoint,device)
    loader=DataLoader(SkeletonDataset(args.manifest,args.split,stream),batch_size=args.batch)
    probs,y,attn=predict(m,loader,device,args.ratio)
    pred=probs.argmax(1);p,r,f,_=precision_recall_fscore_support(y,pred,average='macro',zero_division=0)
    metrics={'accuracy':accuracy_score(y,pred),'macro_precision':p,'macro_recall':r,'macro_f1':f}
    print(json.dumps(metrics,indent=2));np.savez(args.output,probs=probs,labels=y,pred=pred,attention=attn,confusion=confusion_matrix(y,pred,labels=np.arange(51)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--split',default='val');p.add_argument('--batch',type=int,default=64);p.add_argument('--ratio',type=float,default=1);p.add_argument('--output',default='predictions.npz');main(p.parse_args())
