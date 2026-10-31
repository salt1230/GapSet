"""Fixed seed42 weights, predeclared label-blind subset and all 46 conditions."""
import argparse
import hashlib
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main(path):
    p=json.loads(path.read_text())
    out=Path(p['stage'])/'diagnostics'
    out.mkdir(exist_ok=False)
    # Imports use the installed package; no host release directory.
    from gapset.training.guard import Guard
    from gapset.data.dataset import C1Dataset,conditions
    from gapset.models.relative_model import RelativeCSMISSM
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    checkpoint=p['relative_checkpoint']
    assert sha(checkpoint)==p['relative_checkpoint_sha256']
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['recipe']['selection_split']=='source_val_clean' and state['epoch']==176
    model=RelativeCSMISSM().eval()
    model.load_state_dict(state['model'],strict=True)
    before={k:v.clone() for k,v in model.state_dict().items()}
    ds=C1Dataset(p['data'],'cross_domain_dev')
    indices=sorted(sorted(range(len(ds)),key=lambda i:hashlib.sha256(ds.rows[i]['id'].encode()).hexdigest())[:256])
    assert len(indices)==256
    save=lambda name,value:(out/name).write_text(json.dumps(value,indent=2))
    save('selection.json',dict(rule='lowest SHA256 of current development window IDs, no label or score used; then native order',
        n=256,ids=[ds.rows[i]['id'] for i in indices],all_46_conditions=True,not_whole_population_diagnostics=True))
    guard=Guard(out)
    guard.admit()
    model=model.cuda()
    original=[block.integrate_relative for block in model.blocks]
    summaries=[]
    active=[]

    def instrument(index):
        def wrapped(self,value,q,delta,reference_q,reference_delta):
            u=torch.cat([value*q[:,None],q[:,None].expand_as(value),reference_q[:,None].expand_as(value)],1).contiguous()
            dt=torch.cat([delta,delta,reference_delta],1).contiguous()
            a=-self.log_rates.exp()
            c=torch.nn.functional.softplus(self.readout)
            aa=torch.cat([a,a,a]);cc=torch.cat([c,c,c])
            both=selective_scan_fn(u,dt,aa,torch.ones_like(aa),cc)
            numerator,mass,reference_mass=both.chunk(3,1)
            output=torch.where(mass>1e-8,(numerator/mass.clamp_min(1e-8))*reference_mass,0)
            supported=(q[:,None]>0).expand_as(mass)&(mass>1e-8)
            record=active[index]
            record['supported_channel_events']+=int(supported.sum().item())
            record['unsupported_current_events']+=int((q<=0).sum().item())
            for key,value in [('actual_mass',mass),('reference_mass',reference_mass),('correction',reference_mass/mass.clamp_min(1e-8)),('absolute_numerator',numerator.abs()),('absolute_output',output.abs())]:
                values=value[supported]
                if values.numel():
                    target=record[key]
                    target['sum']+=float(values.double().sum().item())
                    target['min']=min(target['min'],float(values.min().item()))
                    target['max']=max(target['max'],float(values.max().item()))
            record['correction_gt100']+=int(((reference_mass/mass.clamp_min(1e-8)>100)&supported).sum().item())
            return output.transpose(1,2)
        return wrapped

    with torch.no_grad():
        for condition in conditions():
            if time.time()+30>=p['deadline_unix']:
                raise TimeoutError('Original post-analysis deadline')
            ds=C1Dataset(p['data'],'cross_domain_dev',condition)
            active[:]=[dict(block=j,supported_channel_events=0,unsupported_current_events=0,correction_gt100=0,
                **{k:dict(sum=0.,min=float('inf'),max=float('-inf')) for k in ['actual_mass','reference_mass','correction','absolute_numerator','absolute_output']}) for j in range(2)]
            all_logits=[];labels=[];parity=0.;qsum=0.;qcount=0;representation_norms=[]
            for start in range(0,len(indices),128):
                guard.check()
                samples=[ds[i] for i in indices[start:start+128]]
                image=torch.stack([r[0] for r in samples]).cuda()
                clock=torch.stack([r[1] for r in samples]).cuda()
                for block,fn in zip(model.blocks,original,strict=True):
                    block.integrate_relative=fn
                reference=model(image,clock)
                for j,block in enumerate(model.blocks):
                    block.integrate_relative=types.MethodType(instrument(j),block)
                actual=model(image,clock)
                parity=max(parity,float((reference-actual).abs().max().item()))
                assert parity<=2e-4
                # Restore for auxiliary representation evaluation, excluding double counts.
                for block,fn in zip(model.blocks,original,strict=True):
                    block.integrate_relative=fn
                _,q=model.tokens(image,clock)
                qsum+=float(q.double().sum().item());qcount+=q.numel()
                representation_norms.extend(model.representation(image,clock).norm(dim=1).cpu().tolist())
                all_logits.append(actual.cpu().numpy());labels.extend(int(r[2]) for r in samples)
            logits=np.concatenate(all_logits)
            y=np.asarray(labels)
            pred=logits.argmax(1)
            cm=np.bincount(y*6+pred,minlength=36).reshape(6,6)
            for record in active:
                n=record['supported_channel_events']
                for key in ('actual_mass','reference_mass','correction','absolute_numerator','absolute_output'):
                    target=record[key]
                    target['mean']=target.pop('sum')/max(n,1)
                    if not n:
                        target['min']=target['max']=None
            summaries.append(dict(condition=condition['name'],challenge=condition,n=len(y),
                class_counts=np.bincount(y,minlength=6).tolist(),confusion=cm.tolist(),
                predicted_class_counts=np.bincount(pred,minlength=6).tolist(),q_mean=qsum/qcount,
                representation_l2_mean=float(np.mean(representation_norms)),blocks=active.copy(),
                instrumented_vs_original_max_logit_difference=parity))
            save('progress.json',dict(completed_conditions=len(summaries),total_conditions=46,time_unix=time.time()))
    for key,value in model.state_dict().items():
        assert torch.equal(value.cpu(),before[key])
    assert sha(checkpoint)==p['relative_checkpoint_sha256']
    save('completion.json',dict(status='complete',rows=summaries,weights_unchanged=True,
        no_training_or_optimizer=True,subset_diagnostics_only=True,not_causal_failure_attribution=True,time_unix=time.time()))
    print(json.dumps(dict(status='complete',conditions=46,windows_each=256,weights_unchanged=True)))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    main(parser.parse_args().plan)
