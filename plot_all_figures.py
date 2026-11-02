"""Figures 4-10 from measured experiment outputs; no fabricated curves."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

PAPER={'baseline':85.72,'joint':86.50,'bone':90.50,'joint_motion':80.60,'bone_motion':74.70,'fused':90.70}
TIMES={'joint':3.6,'bone':4.1,'joint_motion':3.9,'bone_motion':3.5}

def save(fig,out,name):fig.tight_layout();fig.savefig(out/name,dpi=220,bbox_inches='tight');plt.close(fig)
def plot_history(path,out):
    h=json.loads(Path(path).read_text());epochs=[x['epoch'] for x in h]
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,metric,title in zip(axs,['accuracy','loss'],['(a) Accuracy','(b) Loss']):
        for split in ['train','val']:ax.plot(epochs,[x[split][metric] for x in h],label=split)
        ax.set_title(title);ax.set_xlabel('Epoch');ax.set_ylabel(metric.title());ax.legend();ax.grid(alpha=.2)
    save(fig,out,'figure_04_training_curves.png')
def plot_comparison(out,baseline=None,joint=None):
    # Paper reference values are clearly labeled, never called reproduced results
    values=[PAPER['baseline'],PAPER['joint']] if baseline is None or joint is None else [baseline,joint]
    source='Paper-reported reference' if baseline is None or joint is None else 'Measured validation'
    fig,ax=plt.subplots(figsize=(6,4));bars=ax.bar(['Independent baseline','Unified dual-person'],values,color=['#888','#3182bd']);ax.bar_label(bars,fmt='%.2f');ax.set_ylim(0,100);ax.set_ylabel('Accuracy (%)');ax.set_title('Figure 5 — '+source);save(fig,out,'figure_05_model_comparison.png')
def plot_predictions(path,out):
    d=np.load(path);y=d['labels'];pred=d['pred'];n=51
    totals=np.bincount(y,minlength=n);correct=np.bincount(y[y==pred],minlength=n)
    accuracy=np.divide(correct,totals,out=np.full(n,np.nan),where=totals>0)*100
    fig,ax=plt.subplots(figsize=(14,4));ax.bar(np.arange(n),accuracy);ax.set_xlabel('Class ID (0–50)');ax.set_ylabel('Accuracy (%)');ax.set_ylim(0,100);ax.set_title('Figure 6 — Measured per-class accuracy');save(fig,out,'figure_06_per_class_accuracy.png')
    np.savetxt(out/'per_class_accuracy.csv',np.column_stack([np.arange(n),totals,accuracy]),delimiter=',',header='class_id,support,accuracy_percent',comments='')
    cm=confusion_matrix(y,pred,labels=np.arange(n),normalize='true');fig,ax=plt.subplots(figsize=(10,9));im=ax.imshow(cm,cmap='Blues',vmin=0,vmax=1);fig.colorbar(im,ax=ax);ax.set_xlabel('Predicted class');ax.set_ylabel('True class');ax.set_title('Figure 7 — Measured normalized confusion matrix');save(fig,out,'figure_07_confusion_matrix.png')
    np.savetxt(out/'confusion_matrix.csv',cm,delimiter=',')
def plot_attention(predictions,out):
    # Each input is class-specific predictions saved by a separate evaluation subset.
    fig,ax=plt.subplots(figsize=(8,4));has=False
    for name,path in predictions:
        d=np.load(path)
        if 'attention' not in d or d['attention'].size==0:continue
        alpha=d['attention'];ax.plot(np.linspace(0,1,alpha.shape[1]),alpha.mean(0),label=name);has=True
    if has:
        ax.set_xlabel('Normalized temporal position');ax.set_ylabel('Mean attention');ax.set_title('Figure 8 — Measured attention (selected subsets)');ax.legend();save(fig,out,'figure_08_temporal_attention.png')
    else:plt.close(fig)
def plot_partial(path,out):
    with open(path,newline='') as f:rows=list(csv.DictReader(f))
    fig,ax=plt.subplots(figsize=(8,5))
    for name in sorted(set(r['stream'] for r in rows)):
        sub=sorted((r for r in rows if r['stream']==name),key=lambda r:float(r['ratio']))
        ax.plot([100*float(r['ratio']) for r in sub],[100*float(r['accuracy']) for r in sub],marker='o',label=name)
    ax.set_xlabel('Observed prefix (%)');ax.set_ylabel('Accuracy (%)');ax.set_ylim(0,100);ax.set_title('Figure 9 — Measured partial-observation accuracy');ax.legend();save(fig,out,'figure_09_partial_observation.png')
def plot_efficiency(out):
    names=list(TIMES);fig,axs=plt.subplots(1,2,figsize=(12,4));axs[0].bar(names,[1.7]*4);axs[0].set_ylabel('Parameters (millions)');axs[1].bar(names,[TIMES[n] for n in names]);axs[1].set_ylabel('Milliseconds / clip');
    for ax in axs:ax.tick_params(axis='x',rotation=20)
    fig.suptitle('Figure 10 — Paper-reported reference efficiency (not measured)');save(fig,out,'figure_10_paper_reference_efficiency.png')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',default='figures');p.add_argument('--history');p.add_argument('--predictions');p.add_argument('--partial');p.add_argument('--attention',nargs='*',help='name=predictions.npz (class-filtered)');p.add_argument('--baseline-accuracy',type=float);p.add_argument('--joint-accuracy',type=float);a=p.parse_args()
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    if a.history:plot_history(a.history,out)
    plot_comparison(out,a.baseline_accuracy,a.joint_accuracy)
    if a.predictions:plot_predictions(a.predictions,out)
    if a.partial:plot_partial(a.partial,out)
    if a.attention:plot_attention([s.split('=',1) for s in a.attention],out)
    plot_efficiency(out);print('Figures saved to',out)
