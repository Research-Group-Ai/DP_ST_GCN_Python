"""Generate methodological Figures 1-3 (schematic recreations, not original artwork)."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from graph import PARENTS,CROSS

def save(fig,path):
    fig.savefig(path,dpi=220,bbox_inches='tight');plt.close(fig)

def skeleton_graph(out):
    # Kinect-v2 schematic positions (not measured coordinates)
    xy=np.array([[0,0],[0,1],[0,2],[0,3],[-1,1.8],[-2,1.7],[-3,1.5],[-4,1.3],[1,1.8],[2,1.7],[3,1.5],[4,1.3],[-.7,-1],[-.8,-2],[-.8,-3],[-.8,-3.5],[.7,-1],[.8,-2],[.8,-3],[.8,-3.5],[0,2.4],[-4.3,1.5],[-4.5,1.2],[4.3,1.5],[4.5,1.2]],float)
    fig,ax=plt.subplots(figsize=(13,6)); offsets=[-6,6]
    for person,offset in enumerate(offsets):
        p=xy+np.array([offset,0]);color=['#1684bd','#d74a4a'][person]
        for child,parent in enumerate(PARENTS):
            if child!=parent:ax.plot(p[[child,parent],0],p[[child,parent],1],color='#555',lw=1.4,zorder=1)
        ax.scatter(p[:,0],p[:,1],s=32,color=color,zorder=3)
        ax.text(offset,4,'Person '+('A' if person==0 else 'B'),ha='center',fontsize=13)
    for joint in CROSS:
        ax.plot([xy[joint,0]+offsets[0],xy[joint,0]+offsets[1]],[xy[joint,1]]*2,'--',color='#209a6b',lw=1.1,alpha=.7)
    ax.set_title('Figure 1 — Dual-person 50-node interaction graph (schematic)');ax.set_aspect('equal');ax.axis('off');save(fig,out/'figure_01_dual_person_graph.png')

def boxes(ax,labels,title):
    ax.set_xlim(0,10);ax.set_ylim(0,len(labels)*1.25+1);ax.axis('off');ax.set_title(title,fontsize=12)
    for i,label in enumerate(labels):
        y=len(labels)*1.25-i*1.25-.3
        rect=FancyBboxPatch((1,y-.65),8,.85,boxstyle='round,pad=.1',facecolor='#e7f0fa',edgecolor='#376b9b')
        ax.add_patch(rect);ax.text(5,y-.23,label,ha='center',va='center',fontsize=9)
        if i<len(labels)-1:ax.add_patch(FancyArrowPatch((5,y-.75),(5,y-1.12),arrowstyle='-|>',mutation_scale=13,color='#444'))

def architecture(out):
    fig,axs=plt.subplots(1,2,figsize=(12,10))
    boxes(axs[0],['Graph convolution','BatchNorm + ReLU','Dropout (p=0.5)','Temporal convolution (kernel=9)','BatchNorm','Residual addition + ReLU'],'(a) ST-GCN block (residual path omitted visually)')
    boxes(axs[1],['Input: B × 3 × T × 50','ST-GCN 1: 64','ST-GCN 2: 64','ST-GCN 3: 128, stride 2','ST-GCN 4: 128','ST-GCN 5: 256, stride 2','ST-GCN 6: 256','Spatial average + temporal attention','Fully connected → 51 logits'],'(b) DP-ST-GCN backbone')
    save(fig,out/'figure_02_architecture.png')
    fig,ax=plt.subplots(figsize=(11,3));boxes(ax,['Frame-level features h₁ ... hT','eₜ = w₂ᵀ tanh(W₁hₜ + b₁)','α = softmax(e)','z = Σ αₜ hₜ','Dropout + classifier'],'Figure 3 — Learned temporal attention');save(fig,out/'figure_03_attention_diagram.png')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',default='figures');a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);skeleton_graph(out);architecture(out);print('Saved Figures 1–3 to',out)
