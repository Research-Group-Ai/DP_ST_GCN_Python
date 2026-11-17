"""Artificial 51-class inputs for pipeline smoke tests; not a real dataset."""
import json
from pathlib import Path
import numpy as np


def main():
    rng=np.random.default_rng(20261117)
    out=Path('data/demo'); out.mkdir(parents=True,exist_ok=True)
    records=[]
    for split in ['train','val','test']:
        for label in range(51):
            raw=rng.normal(size=(12,2,25,3)).astype(np.float32)*.1
            raw[:,0,:,2]+=2; raw[:,1,:,2]+=2
            raw[:,1,:,0]+=1
            raw[:,:,2,1]+=.5
            name=f'{split}_{label}'
            np.save(out/f'{name}.npy',raw,allow_pickle=False)
            records.append(dict(id=name,path=f'{name}.npy',label=label,split=split,
                                subjects=[f'SYNTHETIC_{split}'],recording=name))
    (out/'manifest.json').write_text(json.dumps(records,indent=2))
    c=json.loads(Path('configs/model.json').read_text())
    c.update(synthetic_demo=True,frames=20,batch_size=17,epochs=1,workers=0)
    Path('configs/demo.json').write_text(json.dumps(c,indent=2))
    print('Artificial smoke-test data created; no real accuracy expected.')


if __name__=='__main__': main()
