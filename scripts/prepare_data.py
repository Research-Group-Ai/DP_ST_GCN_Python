"""Convert explicit recording/split metadata and PKU skeleton/label files to clips."""
import argparse
import json
from pathlib import Path
import numpy as np
from dpstgcn.data import load_manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recordings',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--frame-base',required=True,type=int,choices=[0,1])
    p.add_argument('--minimum-confidence',type=int,default=1,choices=[1,2])
    a=p.parse_args()
    records=json.loads(a.recordings.read_text())
    out=a.output
    out.mkdir(parents=True,exist_ok=True)
    clips=out/'clips'
    clips.mkdir(exist_ok=True)
    manifest=[]
    for i,r in enumerate(records):
        raw=np.loadtxt(a.recordings.parent/r['skeleton'],dtype=np.float32,ndmin=2)
        if raw.shape[1]!=150 or not np.isfinite(raw).all():
            raise ValueError('Skeleton file must have 150 finite floats per frame.')
        raw=raw.reshape(-1,2,25,3)
        label_path=a.recordings.parent/r['labels']
        text=label_path.read_text().strip()
        from io import StringIO
        labels=np.loadtxt(StringIO(text.replace(',', ' ')),ndmin=2)
        # Accept four integer fields separated by commas or whitespace.
        if labels.shape[1]!=4 or not np.equal(labels,labels.astype(int)).all():
            raise ValueError('Labels require four integer fields: class,start,end,confidence.')
        for j,(label,start,end,confidence) in enumerate(labels.astype(int)):
            if confidence not in (1,2):
                raise ValueError('Annotation confidence must be 1 or 2.')
            if confidence<a.minimum_confidence: continue
            start,end=start-a.frame_base,end-a.frame_base
            if not 1<=label<=51 or not 0<=start<=end<len(raw):
                raise ValueError(f'Invalid annotation in recording {i}, row {j}.')
            clip_id=f'recording_{i:05d}_clip_{j:04d}'
            path=clips/f'{clip_id}.npy'
            np.save(path,raw[start:end+1],allow_pickle=False)
            manifest.append(dict(id=clip_id,path=str(path.relative_to(out)),label=int(label-1),
                                 split=r['split'],subjects=r['subjects'],recording=r['recording'],
                                 start_frame=int(start),end_frame=int(end)))
    manifest_path=out/'manifest.json'
    # Audit before writing final manifest; record files remain local if invalid.
    manifest_path.write_text(json.dumps(manifest,indent=2))
    try:
        load_manifest(manifest_path)
    except Exception:
        manifest_path.unlink()
        raise
    (out/'preparation.json').write_text(json.dumps(dict(frame_base=a.frame_base,
       end_frame='inclusive',minimum_confidence=a.minimum_confidence,
       coordinate_layout='person,joint,xyz',recordings=str(a.recordings)),indent=2))
    print(f'Wrote {len(manifest)} clips to {manifest_path}')


if __name__=='__main__': main()
