import torch
from torch import nn
from graph import adjacency

class STBlock(nn.Module):
    def __init__(self,ci,co,stride=1,dropout=.5):
        super().__init__()
        self.spatial=nn.Conv2d(ci,co,1)
        self.bn1=nn.BatchNorm2d(co)
        self.drop=nn.Dropout(dropout)
        self.temporal=nn.Conv2d(co,co,(9,1),stride=(stride,1),padding=(4,0))
        self.bn2=nn.BatchNorm2d(co)
        self.skip=nn.Identity() if ci==co and stride==1 else nn.Sequential(nn.Conv2d(ci,co,1,stride=(stride,1)),nn.BatchNorm2d(co))
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x,a):
        h=torch.einsum('bctv,vw->bctw',x,a)
        h=self.drop(self.relu(self.bn1(self.spatial(h))))
        return self.relu(self.bn2(self.temporal(h))+self.skip(x))

class DPSTGCN(nn.Module):
    def __init__(self,classes=51,dual=True,cross=True,dropout=.5):
        super().__init__()
        self.register_buffer('adj',adjacency(dual,cross))
        channels=[3,64,64,128,128,256,256]
        self.blocks=nn.ModuleList([STBlock(channels[i],channels[i+1],2 if i in (2,4) else 1,dropout) for i in range(6)])
        self.attn=nn.Sequential(nn.Linear(256,64),nn.Tanh(),nn.Linear(64,1))
        self.drop=nn.Dropout(dropout)
        self.fc=nn.Linear(256,classes)
    def forward(self,x,return_attention=False):
        for block in self.blocks: x=block(x,self.adj)
        h=x.mean(-1).transpose(1,2)
        alpha=self.attn(h).squeeze(-1).softmax(-1)
        logits=self.fc(self.drop((h*alpha.unsqueeze(-1)).sum(1)))
        return (logits,alpha) if return_attention else logits

class IndependentBaseline(nn.Module):
    def __init__(self,classes=51):
        super().__init__(); self.backbone=DPSTGCN(classes,dual=False)
    def forward(self,x,return_attention=False):
        a=self.backbone(x[:,:,:,:25]); b=self.backbone(x[:,:,:,25:])
        logits=torch.maximum(a,b)
        if return_attention: return logits,None
        return logits
