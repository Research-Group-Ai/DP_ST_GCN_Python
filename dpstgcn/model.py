"""Six-block graph/temporal network with learned temporal attention."""
import torch
from torch import nn
from .graph import adjacency


class Block(nn.Module):
    def __init__(self,cin,cout,stride=1,dropout=.5):
        super().__init__()
        self.spatial=nn.Sequential(nn.Conv2d(cin,cout,1,bias=False),nn.BatchNorm2d(cout),
                                   nn.ReLU(),nn.Dropout(dropout))
        self.temporal=nn.Sequential(nn.Conv2d(cout,cout,(9,1),stride=(stride,1),padding=(4,0),bias=False),
                                    nn.BatchNorm2d(cout))
        self.residual=nn.Identity() if cin==cout and stride==1 else nn.Sequential(
            nn.Conv2d(cin,cout,1,stride=(stride,1),bias=False),nn.BatchNorm2d(cout))

    def forward(self,x,a):
        spatial=torch.einsum('vw,nctw->nctv',a,x)
        return torch.relu(self.temporal(self.spatial(spatial))+self.residual(x))


class Backbone(nn.Module):
    def __init__(self,classes=51,mode='dual',cross_joints=None,dropout=.5):
        super().__init__()
        self.register_buffer('adjacency',torch.from_numpy(adjacency(mode,cross_joints)))
        channels=[3,64,64,128,128,256,256]
        self.blocks=nn.ModuleList([Block(channels[i],channels[i+1],2 if i in (2,4) else 1,dropout) for i in range(6)])
        self.attention=nn.Sequential(nn.Linear(256,64),nn.Tanh(),nn.Linear(64,1,bias=False))
        self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(256,classes))

    def forward(self,x):
        for block in self.blocks:
            x=block(x,self.adjacency)
        h=x.mean(dim=-1).transpose(1,2)
        alpha=torch.softmax(self.attention(h).squeeze(-1),dim=1)
        z=(h*alpha.unsqueeze(-1)).sum(dim=1)
        return self.head(z),alpha


class Recognizer(nn.Module):
    def __init__(self,config,mode='dual'):
        super().__init__()
        self.mode=mode
        self.backbone=Backbone(config['classes'],'single' if mode=='single' else mode,
                               config['cross_joints'],config['dropout'])

    def forward(self,x):
        if self.mode!='single':
            return self.backbone(x)
        # A shared backbone, one call for both people, with shared BN statistics.
        # Missing-person branch is suppressed at decision pooling.
        n=x.shape[0]
        people=torch.stack((x[:,:,:,:25],x[:,:,:,25:]),dim=1)
        exists=people.abs().sum(dim=(2,3,4))>0
        flat=people.reshape(n*2,*people.shape[2:])
        logits,att=self.backbone(flat)
        logits=logits.reshape(n,2,-1)
        if not exists.any(dim=1).all():
            raise ValueError('An all-zero two-person input cannot be pooled.')
        logits=logits.masked_fill(~exists[...,None],torch.finfo(logits.dtype).min)
        # Paper does not state logits vs probabilities; logits chosen explicitly.
        return logits.max(dim=1).values,att.reshape(n,2,-1)
