"""Explicit single-case commands. Importing this module never launches a run."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import time

def parser():
    from .models.registry import NAMES
    p=argparse.ArgumentParser(description='GapSet-R Widar research code')
    sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare',help='Build a new Widar cache from legally obtained raw data')
    prep.add_argument('--raw',type=Path,required=True);prep.add_argument('--output',type=Path,required=True)
    prep.add_argument('--workers',type=int,choices=[1,2],default=2)
    audit=sub.add_parser('audit',help='Verify complete cache identity and fit-only normalization')
    audit.add_argument('--data',type=Path,required=True);audit.add_argument('--raw-root',type=Path,required=True)
    audit.add_argument('--output',type=Path,required=True)
    masks=sub.add_parser('freeze-masks',help='Freeze all 92 source/development mask digests before scoring')
    masks.add_argument('--data',type=Path,required=True);masks.add_argument('--output',type=Path,required=True)
    means=sub.add_parser('fit-means',help='Fit GRU-D 90-variable means using fit_clean only')
    means.add_argument('--data',type=Path,required=True);means.add_argument('--output',type=Path,required=True)
    for cmd in ['cpu-gate','gate','train','eval']:
        a=sub.add_parser(cmd,help='One explicitly requested case; no queue or search')
        a.add_argument('--model',choices=NAMES,required=True)
        a.add_argument('--seed',type=int,choices=[42,43,44],required=True)
        a.add_argument('--data',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
        a.add_argument('--vendor',type=Path,default=Path('third_party'))
        a.add_argument('--input-identity',type=Path,required=True)
        a.add_argument('--cpu-gate',type=Path)
        a.add_argument('--training',type=Path)
        a.add_argument('--expected-masks',type=Path)
        a.add_argument('--max-hours',type=float,default=24)
        a.add_argument('--lock',type=Path,default=Path('outputs/.gpu.lock'))
    return p

def new_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:json.dump(value,f,indent=2)

def freeze_masks(data,output):
    from .data.dataset import C1Dataset,conditions
    from .data.preprocess import digest
    if output.exists():raise FileExistsError(output)
    rows=[]
    for role in ('source_val_clean','cross_domain_dev'):
        for condition in conditions():
            ds=C1Dataset(data,role,condition);h=hashlib.sha256()
            for i,row in enumerate(ds.rows):
                h.update(row['id'].encode());h.update(ds.observation(i).valid.tobytes())
            rows.append(dict(split=role,condition=condition['name'],mask_sha256=h.hexdigest()))
    new_json(output,dict(tests=rows,input_manifest_sha256=digest(data/'manifest.json'),
                         label_or_model_scores_used=False))

def fit_means(data,output):
    import numpy as np
    from .data.preprocess import digest
    if output.exists():raise FileExistsError(output)
    m=json.loads((data/'manifest.json').read_text())
    if m['status']!='ready':raise ValueError('Unverified cache')
    x=np.memmap(data/'amplitude.bin',mode='r',dtype=np.float32,shape=tuple(m['shape']))
    mask=np.memmap(data/'valid.bin',mode='r',dtype=bool,shape=tuple(m['shape']))
    total=np.zeros((3,30),dtype=np.float64);count=np.zeros((3,30),dtype=np.int64)
    rows=[r for r in m['records'] if r['role']=='fit_clean']
    for r in rows:
        a,v=x[r['row']],mask[r['row']]
        z=(np.log1p(np.where(v,a,0)).astype(np.float32)-m['normalizer']['mean'])/m['normalizer']['std']
        total+=np.where(v,z,0).sum(0,dtype=np.float64);count+=v.sum(0)
    if not (count>0).all():raise ValueError('Unobserved fit variable')
    new_json(output,dict(role='fit_clean',records=len(rows),means_normalized_csi=(total/count).reshape(-1).tolist(),
       counts=count.reshape(-1).tolist(),no_validation_or_development_used=True,manifest_sha256=digest(data/'manifest.json')))

def cpu_gate(args,worker):
    import torch
    from .models.registry import make_model
    from .data.dataset import C1Dataset
    from .data.preprocess import digest
    if args.output.exists():raise FileExistsError(args.output)
    identity=json.loads(args.input_identity.read_text())
    if identity['status']!='passed' or identity['manifest_sha256']!=digest(args.data/'manifest.json'):
        raise ValueError('Input identity mismatch')
    ds=C1Dataset(args.data,'fit_clean');samples=[ds[0],ds[len(ds)-1]]
    image=torch.stack([r[0] for r in samples]);ts=torch.stack([r[1] for r in samples])
    labels=torch.tensor([r[2] for r in samples])
    worker.preimport(args.model);worker.seed();model=make_model(args.model,args.vendor)
    initial=worker.model_fingerprint(model)
    worker.seed();again=make_model(args.model,args.vendor)
    if worker.model_fingerprint(again)!=initial:raise ValueError('Reinitialization mismatch')
    del again
    output=model(image,ts);torch.nn.functional.cross_entropy(output,labels).backward()
    missing=[n for n,v in model.named_parameters() if v.grad is None]
    if missing and args.model not in ('csibert2_clean_backbone','dfs_csi_backbone'):
        raise ValueError('Missing model gradients: '+str(missing))
    if not all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None):
        raise ValueError('Nonfinite gradients')
    model.eval()
    with torch.no_grad():error=(model(image,ts)-torch.cat([model(image[i:i+1],ts[i:i+1]) for i in range(2)])).abs().max().item()
    if error>=2e-4 or torch.cuda.is_initialized():raise ValueError('CPU parity/device gate')
    args.output.mkdir(parents=True)
    new_json(args.output/'cpu_gate.json',dict(status='passed',cuda_used=False,model=args.model,seed=args.seed,
      initial_model_sha256=initial,params=sum(v.numel() for v in model.parameters()),
      batch_parity_maxdiff=error,dormant_parameters=missing))

@contextlib.contextmanager
def single_gpu_lock(path):
    import fcntl
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)

def main():
    p=parser();a=p.parse_args()
    if a.command=='prepare':
        from .data.preprocess import prepare
        return prepare(a.raw,a.output,a.workers)
    if a.command=='audit':
        from .data.audit import audit
        if a.output.exists():raise FileExistsError(a.output)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        return audit(a.data,a.output,a.raw_root)
    if a.command=='freeze-masks':return freeze_masks(a.data,a.output)
    if a.command=='fit-means':return fit_means(a.data,a.output)
    from .training import worker
    from .training.guard import ResourcePause
    worker.SEED=a.seed
    if a.command=='cpu-gate':return cpu_gate(a,worker)
    if not 0<a.max_hours<=168:p.error('--max-hours must be in (0,168]')
    if a.cpu_gate is None:p.error('--cpu-gate is required for gate/train/eval')
    cpu=json.loads(a.cpu_gate.read_text())
    if cpu['model']!=a.model or cpu['seed']!=a.seed:p.error('CPU gate belongs to another model/seed')
    if a.command=='eval':
        if a.training is None or a.expected_masks is None:p.error('eval requires --training and --expected-masks')
        from .data.preprocess import digest
        masks=json.loads(a.expected_masks.read_text())
        if masks['input_manifest_sha256']!=digest(a.data/'manifest.json'):p.error('Frozen masks belong to another cache')
        import torch
        state=torch.load(a.training/'best.pth',map_location='cpu',weights_only=False)
        if state['recipe']['model']!=a.model or state['recipe']['seed']!=a.seed:p.error('Checkpoint model/seed mismatch')
        del state
    os.environ['GAPSET_RUN_DEADLINE']=str(time.time()+a.max_hours*3600)
    with single_gpu_lock(a.lock):
        try:return {'gate':worker.gate,'train':worker.train,'eval':worker.evaluate}[a.command](a)
        except ResourcePause as exc:
            print(str(exc));raise SystemExit(75)

if __name__=='__main__':main()
