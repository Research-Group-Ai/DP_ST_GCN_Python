import argparse,numpy as np
from sklearn.metrics import accuracy_score
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--joint',required=True);p.add_argument('--bone',required=True);p.add_argument('--output',default='fused.npz');a=p.parse_args()
    j=np.load(a.joint);b=np.load(a.bone)
    if not np.array_equal(j['labels'],b['labels']):raise ValueError('Sample order/labels mismatch')
    y=j['labels'];best=(-1,None,None)
    for lam in np.linspace(0,1,101):
        probs=lam*j['probs']+(1-lam)*b['probs'];acc=accuracy_score(y,probs.argmax(1))
        if acc>best[0]:best=(acc,lam,probs)
    print(f'Validation accuracy={best[0]:.4f}; joint weight={best[1]:.2f}')
    np.savez(a.output,probs=best[2],labels=y,pred=best[2].argmax(1),joint_weight=best[1])
