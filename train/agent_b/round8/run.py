"""R8 paired starter benchmark. CUDA is leased one fold at a time."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from train.agent_b.round4.run import read, sha, write
from train.shared.artifacts import Experiment, aligned_predictions
from train.shared.data import load_dataset, split_indices
from train.shared.metrics import evaluate, score_mask
from train.shared.preprocess import Normalizer, temporal_windows
from .control import ORIG, ROUND, ROOT, Pending, artifact, protocol, source, starter
from .mechanisms import Readout, batches, weights


ARMS={
    'B1-60':dict(route='B1',half=60), 'B1-30':dict(route='B1',half=30), 'B1-90':dict(route='B1',half=90),
    **{f'B2-{w}-{n}':dict(route='B2',days=w,steps=n) for w in (0,30,60) for n in (100,300)},
    'B3-shared':dict(route='B3',steps=300,penalty=0),
    'B3-local-low':dict(route='B3',steps=300,penalty=.10),
    'B3-local-high':dict(route='B3',steps=300,penalty=1.),
}


def control(p,digest,cut,seed):
    doc=read(Path(p['paths']['controls']))
    if doc['protocol_sha256']!=digest: raise ValueError('Controls protocol mismatch')
    row=doc['controls'].get(cut,{}).get(str(seed))
    if row is None: raise Pending(f'No A control for {cut}/s{seed}')
    return row


def inputs(ds,norm,stop):
    w,m=temporal_windows(ds.x[:stop],ds.times[:stop],(2,1,0)); h=ds.times[:stop].hour.to_numpy(dtype=np.int64)
    return tuple(torch.from_numpy(a) for a in (norm.inputs(w),h,m))


def links(row):
    return {k:artifact(row[k],ROUND) for k in ('checkpoint','normalizer','prediction')}


def compare_alignment(ds,va,bounds,path):
    meta,(arr,)=aligned_predictions([path]); t=ds.times[va]
    for k,v in [('times',t.strftime('%Y%m%d%H')),('scored',score_mask(t)),
                ('wrf_lats',ds.static['wrf_lats']),('wrf_lons',ds.static['wrf_lons'])]: np.testing.assert_array_equal(meta[k],v)
    if (meta['start'].item(),meta['end_exclusive'].item())!=bounds: raise ValueError('Fold mismatch')
    return arr


def model_for(ds,spec,ck):
    model=starter(ds.static)
    if spec['route']!='B1': model.load_state_dict(ck.get('state_dict',ck))
    if spec['route']=='B2': model.requires_grad_(False); model.dec.requires_grad_(True)
    if spec['route']=='B3': model=Readout(model,spec['penalty']>0)
    return model


@torch.inference_mode()
def predict(model,x,ix,norm):
    model.eval()
    pred=np.concatenate([model(*(a[ix[s:s+128]] for a in x)).cpu().numpy() for s in range(0,len(ix),128)])
    return norm.physical(pred,x[1][ix].cpu().numpy())


def cuda_fit(ds,norm,tr,va,spec,seed,ck,p):
    cutoff=pd.Timestamp(p['folds'][p['_cut']]['start'],tz='UTC'); t=ds.times[tr]
    x=inputs(ds,norm,va[-1]+1); y=torch.from_numpy(norm.targets(ds.y[tr],t.hour.to_numpy()))
    gpu=source('r8_gpu',ORIG/'train/agent_a/gpu.py'); tick=perf_counter()
    extra={}; weighted=spec['route']=='B1'
    if weighted: w,extra=weights(t,cutoff,spec['half'])
    with gpu.gpu_lock(Path(p['paths']['gpu_lock'])) as wait:
        if datetime.now(timezone.utc)>=datetime.fromisoformat(p['deadlines']['confirmation_end_utc']):
            raise Pending('Training cutoff reached while awaiting GPU')
        torch.manual_seed(seed); torch.cuda.reset_peak_memory_stats()
        model=model_for(ds,spec,ck).to('cuda'); x=tuple(a.to('cuda') for a in x); y=y.to('cuda')
        if weighted: w=torch.from_numpy(w).to('cuda')
        params=[v for v in model.parameters() if v.requires_grad]
        opt=torch.optim.AdamW(params,lr=.002 if weighted else .0001,weight_decay=.0001)
        if weighted:
            count=(len(tr)+31)//32; steps=21*count
            sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.002,total_steps=25*count)
        else: steps=spec['steps']
        stream=batches(len(tr),seed,times=t,cutoff=cutoff,days=spec.get('days') or None)
        for step in range(steps):
            if step%32==0 and datetime.now(timezone.utc)>=datetime.fromisoformat(p['deadlines']['confirmation_end_utc']):
                raise Pending('Training cutoff reached; no incomplete artifact exported')
            j=next(stream); model.train(); opt.zero_grad(set_to_none=True)
            out=model(*(a[tr[j]] for a in x)); target=y[j]
            obj=((out-target).square().mean((1,2,3))*w[j]).mean() if weighted else F.mse_loss(out,target)
            if spec['route']=='B3': obj=obj+spec['penalty']*model.penalty()
            obj.backward(); opt.step()
            if weighted: sch.step()
        pred=predict(model,x,va,norm)
        ckout={k:v.detach().cpu() for k,v in model.state_dict().items()}
        extra.update(updates=steps,trainable_parameters=sum(v.numel() for v in params),total_parameters=sum(v.numel() for v in model.parameters()),
                     gpu_peak_mib=torch.cuda.max_memory_allocated()/2**20,lock_wait_seconds=wait)
        del model,x,y,opt,params,obj,out,target
        if weighted: del w
        torch.cuda.synchronize(); torch.cuda.empty_cache()
    return pred,ckout,dict(**extra,seconds=perf_counter()-tick)


def report_fold(ds,va,norm,pred,base,start):
    t=ds.times[va]; y=ds.y[va]; clim=norm.clim[t.hour.to_numpy()]
    pred,base=pred.copy(),base.copy()
    for a in (pred,base): np.clip(a[...,0],0,100,out=a[...,0])
    day=np.asarray((t.normalize()-pd.Timestamp(start,tz='UTC')).days); weeks=[]
    for j in np.unique(day//7):
        k=day//7==j
        weeks.append([evaluate(a[k],y[k],t[k],clim[k]) for a in (pred,base)])
    mask=score_mask(t)
    diagnostics={}
    for name,k in [('all',np.ones(y.shape[:-1],dtype=bool)),('cold',y[...,1]<clim[...,1]),('dry',y[...,0]<clim[...,0])]:
        k=k&mask[:,None,None]
        if k.any(): diagnostics[name]=[dict(count=int(k.sum()),bias=(a-y)[k].mean(0).tolist(),rmse=np.sqrt(((a-y)[k]**2).mean(0)).tolist()) for a in (pred,base)]
    return dict(candidate=evaluate(pred,y,t,clim),control=evaluate(base,y,t,clim),weeks=weeks,diagnostics=diagnostics)


def run(arm,cut,seed,output=None):
    p,digest=protocol(); row=control(p,digest,cut,seed); paths=links(row)
    b=p['folds'][cut]; bounds=(b['start'],b['end_exclusive']); ds=load_dataset(Path(p['paths']['data']))
    tr,va=split_indices(ds.times,*bounds); hs=ds.times.hour.to_numpy(); last=ds.times[tr[-1]].isoformat()
    # A must attest epoch, seed and training provenance; do not infer these from filenames.
    if (row.get('epoch')!=21 or row.get('onecycle_epochs')!=25 or row.get('train_end_exclusive')!=bounds[0]
            or row.get('seed')!=seed or row.get('train_hours')!=len(tr)):
        raise ValueError(f'Control does not attest pre-cut epoch 21: {sorted(row)}')
    with np.load(paths['normalizer'],allow_pickle=False) as z: norm=Normalizer(**{k:z[k] for k in ('x_mean','x_sd','clim','anom_sd')})
    check=Normalizer.fit(ds.x[tr],ds.y[tr],hs[tr]);
    for k,v in vars(check).items(): np.testing.assert_array_equal(getattr(norm,k),v)
    base=compare_alignment(ds,va,bounds,paths['prediction']); ck=torch.load(paths['checkpoint'],weights_only=True,map_location='cpu')
    cfg=ck.get('spec',{})
    # A records EMA shadows alongside the online control; `main` is saved
    # before copying any shadow into the model, and never trains against it.
    if (ck.get('epoch')!=21 or ck.get('seed')!=seed or ck.get('weights')!='main'
            or cfg.get('ridge','missing') is not None or cfg.get('aux')!=0.):
        raise ValueError('Expected unmodified A starter checkpoint')
    if abs(evaluate(base,ds.y[va],ds.times[va],norm.clim[hs[va]])['official']-row['metrics']['raw']['official'])>1e-10:
        raise ValueError('A control rescore mismatch')
    dest=output or ROOT/'runs'/arm/f's{seed}'/cut
    if (dest/'summary.json').exists():
        result=read(dest/'summary.json')
        for k in ('prediction','checkpoint','normalizer'): artifact(result[k],dest)
        verify=compare_alignment(ds,va,bounds,dest/result['prediction']['path'])
        assert result['control_source']==row and result['evaluation']==report_fold(ds,va,norm,verify,base,bounds[0])
        return result
    spec=ARMS[arm]; pred,model,resources=cuda_fit(ds,norm,tr,va,spec,seed,ck,dict(p,_cut=cut))
    if dest.exists(): dest.rename(dest.with_name(f'{dest.name}.partial-{datetime.now(timezone.utc).timestamp():.6f}'))
    out=Experiment(dest); t=ds.times[va]; out.predictions('prediction',pred,t,ds.static,score_mask(t),*bounds)
    torch.save(dict(state_dict=model,config=spec,seed=seed,train_end_exclusive=bounds[0]),dest/'best.pt')
    np.savez(dest/'normalizer.npz',**vars(norm))
    def link(name): return dict(path=name,sha256=sha(dest/name))
    replay=compare_alignment(ds,va,bounds,dest/'prediction.npz'); np.testing.assert_array_equal(replay,pred)
    result=dict(arm=arm,configuration=spec,cut=cut,seed=seed,train_end_exclusive=bounds[0],train_last=last,
                protocol_sha256=digest,prediction_kind='full',control_source=row,resources=resources,
                prediction=link('prediction.npz'),checkpoint=link('best.pt'),normalizer=link('normalizer.npz'),
                evaluation=report_fold(ds,va,norm,replay,base,bounds[0]),
                command=f'python -B -m train.agent_b.round8.run {arm} {cut} --seed {seed}')
    out.json('summary',result); return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('arm',choices=ARMS); parser.add_argument('cut',choices=('C0','C1','C2'))
    parser.add_argument('--seed',type=int,choices=(0,1,2),default=0); parser.add_argument('--output',type=Path)
    args=parser.parse_args(); torch.set_num_threads(2); print(run(args.arm,args.cut,args.seed,args.output)['evaluation']['candidate'])
