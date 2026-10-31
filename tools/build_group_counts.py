"""Build paired gesture-group CM arrays from explicit local evaluation outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--evaluations',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    manifest=json.loads(a.manifest.read_text())
    meta=[r for r in manifest['records'] if r['role']=='cross_domain_dev']
    y=np.asarray([r['label'] for r in meta]);groups=sorted({r['group_id'] for r in meta})
    index={g:i for i,g in enumerate(groups)};gi=np.asarray([index[r['group_id']] for r in meta])
    if np.bincount(gi).max()>=65535:raise ValueError('Group count exceeds uint16')
    users=np.full(len(groups),-1,dtype=int)
    for r in meta:
        g=index[r['group_id']]
        if users[g] not in (-1,r['user']):raise ValueError('Group spans users')
        users[g]=r['user']
    records=[]
    for file in a.evaluations:
        report=json.loads(file.read_text())
        if report['status']!='complete':raise ValueError('Incomplete evaluation')
        records.extend(r for r in report['rows'] if r['split']=='cross_domain_dev')
    names=sorted({r['model'] for r in records});seeds=sorted({r['training_seed'] for r in records})
    conditions=sorted({r['condition'] for r in records})
    if seeds!=[42,43,44] or len(conditions)!=46 or 'clean' not in conditions:raise ValueError('Requires all three seeds and 46 conditions')
    if len(records)!=len(names)*3*46:raise ValueError('Incomplete/duplicate case panel')
    counts=np.zeros((len(names),3,46,len(groups),6,6),dtype=np.uint16);seen=set()
    for r in records:
        key=(r['model'],r['training_seed'],r['condition'])
        if key in seen:raise ValueError('Duplicate condition')
        seen.add(key)
        if r['input_manifest_sha256']!=digest(a.manifest):raise ValueError('Input identity mismatch')
        for field in ('label_file','prediction_file','logit_file'):
            if digest(r[field])!=r[field+'_sha256']:raise ValueError('Prediction checksum mismatch')
        target=np.load(r['label_file'],allow_pickle=False);pred=np.load(r['prediction_file'],allow_pickle=False)
        logits=np.load(r['logit_file'],allow_pickle=False)
        if logits.shape!=(len(y),6) or not np.isfinite(logits).all():raise ValueError('Invalid logits')
        if not np.array_equal(target,y) or not np.array_equal(pred,logits.argmax(1)):raise ValueError('Prediction order/argmax mismatch')
        grouped=np.bincount(gi*36+y*6+pred,minlength=len(groups)*36).reshape(len(groups),6,6)
        if not np.array_equal(grouped.sum(0),r['confusion']):raise ValueError('CM mismatch')
        counts[names.index(key[0]),seeds.index(key[1]),conditions.index(key[2])]=grouped
    a.output.parent.mkdir(parents=True,exist_ok=True)
    # File handle prevents numpy from silently adding a suffix to the requested path.
    with a.output.open('xb') as f:
        np.savez_compressed(f,counts=counts,group_users=users,models=np.asarray(names),
             seeds=np.asarray(seeds),conditions=np.asarray(conditions),group_ids=np.asarray(groups))

if __name__=='__main__':main()
