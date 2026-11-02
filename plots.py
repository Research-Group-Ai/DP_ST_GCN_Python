import argparse,json,numpy as np,matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--predictions');p.add_argument('--history');a=p.parse_args()
    if a.history:
        h=json.load(open(a.history));plt.plot([x['epoch'] for x in h],[x['train']['accuracy'] for x in h],label='Train');plt.plot([x['epoch'] for x in h],[x['val']['accuracy'] for x in h],label='Validation');plt.xlabel('Epoch');plt.ylabel('Accuracy');plt.legend();plt.tight_layout();plt.savefig('learning_curve.png',dpi=180);plt.close()
    if a.predictions:
        d=np.load(a.predictions);cm=confusion_matrix(d['labels'],d['pred'],labels=np.arange(51),normalize='true');fig,ax=plt.subplots(figsize=(11,10));ax.imshow(cm,cmap='Blues',vmin=0,vmax=1);ax.set_xlabel('Predicted');ax.set_ylabel('True');fig.tight_layout();fig.savefig('confusion_matrix.png',dpi=180);plt.close(fig)
        if 'attention' in d and d['attention'].size:
            plt.plot(d['attention'].mean(0));plt.xlabel('Downsampled frame');plt.ylabel('Mean attention');plt.tight_layout();plt.savefig('attention.png',dpi=180)
