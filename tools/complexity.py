"""Full CPU synthetic forward FLOPs; no checkpoint/data, no latency claims."""
import argparse
import collections
import json
import math
from pathlib import Path
NAMES=['sensefi_mlp','sensefi_lenet','sensefi_resnet18','sensefi_lstm','grud_common8','sensefi_vit',
 'sensefi_cnn_gru','wiflexformer','csibert2_clean_backbone','hms_rwkv_full','amcs_tcn_c1','dfs_csi_backbone','gapset_relative']
EXPECTED=[63047558,13576262,11182109,78726,120634,138326,140012,41115,5534023,65942,239115,636774,73100]
VENDOR=Path('third_party')

def run(name, suffix=''):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled=False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(20261022)
    from gapset.models.registry import make_model
    model=make_model('ours_gapset_relative_mass_control' if name=='gapset_relative' else name, VENDOR)
    model.eval()
    # Random initial weights only. Fully observed synthetic common8 native grid.
    x=torch.randn(1,8,30,256)
    x[:,3:6]=1
    timestamp=torch.arange(256,dtype=torch.float32)[None]*.01
    x[:,6]=timestamp[:,None,:]
    x[:,7]=torch.linspace(-1,1,30)[None,:,None]
    if suffix:
        x[:,3:6,:15,::2]=0
        x[:,3:6,:,1::3]=0
        x[:,:3]*=x[:,3:6]

    def tensors(v):
        if isinstance(v,torch.Tensor):return [v]
        if isinstance(v,(tuple,list)):return sum((tensors(i) for i in v),[])
        if isinstance(v,dict):return sum((tensors(i) for i in v.values()),[])
        return []
    def num(v):return math.prod(v.shape)
    def floating(v):return isinstance(v,torch.Tensor) and v.is_floating_point()

    class Counter(TorchDispatchMode):
        def __init__(self):
            super().__init__();self.count=collections.Counter();self.calls=collections.Counter();self.zero=collections.Counter();self.unknown={};self.shapes={}
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            kwargs=kwargs or {};out=func(*args,**kwargs)
            key=str(func);op=key.split('.')[1]
            outs=[a for a in tensors(out) if floating(a)]
            inputs=[a for a in tensors(args) if floating(a)]
            n=num(outs[0]) if outs else 0;v=0;known=True
            if not outs or not inputs: pass
            elif op in ['mm','bmm']: v=2*n*args[0].shape[-1]
            elif op in ['addmm','baddbmm']:
                v=2*n*args[1].shape[-1]+(n if kwargs.get('beta',1)!=0 else 0)
            elif op in ['convolution','_convolution']:
                inp,weight,bias=args[:3];transposed=args[6];groups=args[8]
                if transposed: v=2*num(inp)*weight.shape[1]*math.prod(weight.shape[2:])
                else: v=2*n*weight.shape[1]*math.prod(weight.shape[2:])
                if bias is not None:v+=n
            elif op in ['add','sub','rsub','mul','div','pow','neg','reciprocal','exp','log','sin','cos','sqrt','rsqrt','tanh','sigmoid','gelu','silu','softplus','expm1','log1p']:
                v=n
                if op in ['add','sub'] and kwargs.get('alpha',1)!=1:v+=n
            elif op in ['sum','mean']:
                v=max(0,num(inputs[0])-n)+(n if op=='mean' else 0)
            elif op in ['var','var_mean']:
                # one mean, centered square, sum and division, per output group.
                v=4*num(inputs[0])
            elif op in ['_softmax','_log_softmax']:
                dim=args[1];g=n//args[0].shape[dim]
                v=4*n-g  # subtract max, exp, reduction, divide; max comparisons excluded.
                if op=='_log_softmax':v+=n
            elif op in ['native_layer_norm','native_group_norm']:
                if op=='native_layer_norm':g=num(out[1]);weight,bias=args[2:4]
                else:g=num(out[1]);weight,bias=args[1:3]
                v=6*n+2*g+(n if weight is not None else 0)+(n if bias is not None else 0)
            elif op in ['native_batch_norm','_native_batch_norm_legit_no_training','_native_batch_norm_legit']:
                inp,weight,bias=args[:3]
                v=2*n+2*inp.shape[1]+(n if weight is not None else 0)+(n if bias is not None else 0)
            elif op in ['_adaptive_avg_pool2d','adaptive_avg_pool2d']:
                inp=args[0];oh,ow=outs[0].shape[-2:];ih,iw=inp.shape[-2:]
                area=sum(math.ceil((i+1)*ih/oh)-math.floor(i*ih/oh) for i in range(oh))*sum(math.ceil((j+1)*iw/ow)-math.floor(j*iw/ow) for j in range(ow))
                v=math.prod(inp.shape[:-2])*area
            elif op in ['avg_pool1d','avg_pool2d','avg_pool3d']:
                k=args[1];k=[k] if isinstance(k,int) else k;v=n*math.prod(k)
            elif op.startswith('_scaled_dot_product'):
                q,k,val=args[:3];l,s=q.shape[-2],k.shape[-2];b=math.prod(q.shape[:-2]);d=q.shape[-1];dv=val.shape[-1]
                v=2*b*l*s*(d+dv)+b*l*s+4*b*l*s-b*l
                mask=kwargs.get('attn_mask',args[5] if len(args)>5 else None)
                if mask is not None and mask.is_floating_point():v+=b*l*s
            elif op in ['add_','sub_','mul_','div_','pow_','sigmoid_','tanh_']:v=n
            elif op in ['view','_unsafe_view','reshape','transpose','permute','t','slice','select','unsqueeze','squeeze',
                        'expand','as_strided','detach','alias','clone','contiguous','copy_','_to_copy','to','lift_fresh',
                        'cat','stack','split','unsafe_split','split_with_sizes','unbind','chunk','constant_pad_nd','reflection_pad1d','embedding',
                        'gather','index','index_select','index_put_','masked_fill','masked_fill_','where','flip','repeat',
                        'relu','relu_','clamp','clamp_min','clamp_max','clamp_','threshold','maximum','minimum','max','min',
                        'amax','amin','cummax','cummin','any','all','eq','ne','gt','ge','lt','le','bitwise_and','bitwise_or',
                        'logical_and','logical_or','logical_not','zeros_like','ones_like','empty_like','new_zeros','new_empty',
                        'new_ones','empty','empty_strided','zeros','ones','full','full_like','arange','scalar_tensor',
                        'fill_','zero_','triu','tril','max_pool2d_with_indices','max_pool1d','max_pool2d','dropout',
                        'feature_dropout','native_dropout','isfinite','isnan','isinf','resolve_conj','resolve_neg','sym_size',
                        'sym_numel','size','numel','item','_local_scalar_dense','floor_divide']:
                pass
            else:
                known=False
                self.unknown.setdefault(key,{'args':[list(a.shape) for a in inputs], 'out':[list(a.shape) for a in outs]})
            if known:
                self.count[key]+=int(v);self.calls[key]+=1
                if v==0:self.zero[key]+=1
            return out

    with torch.no_grad():
        baseline=model(x,timestamp)
        counter=Counter()
        with counter: actual=model(x,timestamp)
    delta=float((baseline-actual).abs().max())
    assert delta==0 and actual.shape==(1,6) and torch.isfinite(actual).all()
    params=sum(p.numel() for p in model.parameters())
    assert params==EXPECTED[NAMES.index(name)],(name,params)
    record={'model':name,'parameters_registered':params,'input_shape':[1,8,30,256],'timestamp_shape':[1,256],
            'synthetic_only':True,'checkpoint_loaded':False,'device':'cpu','torch_version':torch.__version__,
            'counted_flops':sum(counter.count.values()),'operator_flops':dict(counter.count),'operator_calls':dict(counter.calls),
            'zero_cost_ops':dict(counter.zero),'unknown':counter.unknown,'instrumentation_max_logit_difference':delta,
            'mask':'subcarrier_and_full_time_deletion' if suffix else 'fully_observed',
            'scope':'full_model_algorithmic_forward_not_hardware_latency'}
    record['accounting_complete']=not bool(counter.unknown)
    return record

def main():
    global VENDOR
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=NAMES,required=True)
    p.add_argument('--vendor',type=Path,default=Path('third_party'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--masked',action='store_true');a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    VENDOR=a.vendor
    result=run(a.model,'_masked' if a.masked else '')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    if not result['accounting_complete']:raise SystemExit('Incomplete FLOPs: unknown operators are recorded')

if __name__=='__main__':main()
