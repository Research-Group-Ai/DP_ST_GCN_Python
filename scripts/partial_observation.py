"""Evaluate prefixes, truncating raw clips before all preprocessing."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--device',default='cpu')
    p.add_argument('--fractions',nargs='+',type=float,default=[.1,.2,.3,.4,.5,.6,.7,.8,.9,1.])
    a=p.parse_args(); rows=[]
    for fraction in a.fractions:
        out=a.output/f'fraction_{fraction:.2f}'
        subprocess.run([sys.executable,'scripts/evaluate.py','--checkpoint',a.checkpoint,
          '--manifest',a.manifest,'--split',a.split,'--fraction',str(fraction),
          '--device',a.device,'--output',str(out)],check=True)
        m=json.loads((out/'metrics.json').read_text()); m['fraction']=fraction; rows.append(m)
    (a.output/'partial_observation.json').write_text(json.dumps(rows,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,4))
    ax.plot([r['fraction']*100 for r in rows],[r['accuracy']*100 for r in rows],marker='o')
    ax.set(xlabel='Observed action prefix (%)',ylabel='Classification accuracy (%)')
    fig.tight_layout(); fig.savefig(a.output/'partial_observation.png',dpi=180)


if __name__=='__main__': main()
