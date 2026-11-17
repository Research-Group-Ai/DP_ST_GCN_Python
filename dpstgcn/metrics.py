import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix


def save_metrics(labels,probabilities,output,classes=51):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    pred=np.asarray(probabilities).argmax(1)
    p,r,f,s=precision_recall_fscore_support(labels,pred,labels=np.arange(classes),zero_division=0)
    # Explicit all-class macro convention; unsupported classes contribute zero.
    metrics=dict(accuracy=float(accuracy_score(labels,pred)),macro_precision=float(p.mean()),
                 macro_recall=float(r.mean()),macro_f1=float(f.mean()),n=len(labels))
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2))
    np.savetxt(output/'per_class.csv',np.column_stack([np.arange(classes),p,r,f,s]),delimiter=',',
               header='class_id,precision,recall,f1,support',comments='')
    cm=confusion_matrix(labels,pred,labels=np.arange(classes))
    normalized=np.divide(cm,cm.sum(1,keepdims=True),out=np.zeros_like(cm,dtype=float),where=cm.sum(1,keepdims=True)>0)
    np.savetxt(output/'confusion_counts.csv',cm,delimiter=',',fmt='%d')
    np.savetxt(output/'confusion_normalized.csv',normalized,delimiter=',')
    print(json.dumps(metrics))
    return metrics
