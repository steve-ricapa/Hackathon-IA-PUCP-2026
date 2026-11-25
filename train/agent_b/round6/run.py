"""Chronological wind specialist benchmark; development only."""
import argparse
from contextlib import nullcontext
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from train.agent_b.round4.run import ORIG, pool, read, sha, write
from train.agent_b.round5.loop import Development, ROOT as R5
from train.shared.artifacts import Experiment, aligned_predictions
from train.shared.geo import bilinear
from train.shared.metrics import OFFICIAL_SCALE, evaluate, score_mask
from train.shared.preprocess import temporal_windows
from .model import WindNet, loss
from .physics import features


R6=ORIG/'.private/experiments/round6'
ROOT=R6/'agent_b'
SPECS={
    'uv':('uv',False,'local',0.,0.),
    'polar':('polar',False,'local',0.,0.),
    'hybrid':('hybrid',False,'local',0.,0.),
    'physics':('uv',True,'local',0.,0.),
    'pyramid-control':('uv',False,'pyramid',0.,0.),
    'pyramid-physics':('uv',True,'pyramid',0.,0.),
    'polar-physics':('polar',True,'local',0.,0.),
    'hybrid-physics':('hybrid',True,'local',0.,0.),
    'polar-aux':('polar',True,'local',.10,0.),
    'spectral':('uv',True,'spectral',0.,0.),
    'extreme':('uv',True,'local',0.,.4),
    'angle':('angle',True,'local',0.,0.),
}


def protocol():
    p=R6/'protocol.json'; digest=sha(p); result=read(p)
    if digest!=(R6/'READY').read_text().strip() or result['development']!=['C0','C1','C2']:
        raise ValueError('Unverified protocol')
    if Path(result['paths']['gpu_lock']).resolve()!=(ORIG/'.private/gpu.lock').resolve():
        raise ValueError('Unexpected lock path')
    if result['metric']['d']!=OFFICIAL_SCALE.tolist(): raise ValueError('Unexpected metric')
    return result,digest


def lease(protocol_):
    spec=spec_from_file_location('shared_cuda_lease',ORIG/'train/agent_a/gpu.py')
    module=module_from_spec(spec); spec.loader.exec_module(module)
    return module.gpu_lock(Path(protocol_['paths']['gpu_lock']))


def assert_reference(meta,d,cut):
    _,_,_,t,bounds,*_=d.fold(cut)
    for key,arr in [('times',t.strftime('%Y%m%d%H')),('scored',score_mask(t)),
                    ('wrf_lats',d.ds.static['wrf_lats']),('wrf_lons',d.ds.static['wrf_lons'])]:
        np.testing.assert_array_equal(meta[key],arr)
    assert (meta['start'].item(),meta['end_exclusive'].item())==bounds


def reference(d,cut,r5):
    link=r5['experiments']['ensemble-05-1']['cuts'][cut]['prediction']; p=R5/link['path']
    assert sha(p)==link['sha256']
    meta,(base,)=aligned_predictions([p]); assert_reference(meta,d,cut)
    return base


def input_fields(ds,stop):
    w,missing=temporal_windows(ds.x[:stop],ds.times[:stop],(2,1,0))
    grids=[bilinear(w[:,i],ds.static) for i in range(3)]; current=grids[-1]
    static,physics=features(ds.static,current)
    h=ds.times[:stop].hour.to_numpy()*np.pi/12
    tags=np.column_stack([np.sin(h),np.cos(h),missing]).astype('float32')
    tags=torch.from_numpy(tags)[:,:,None,None].expand(-1,-1,*current.shape[1:3])
    core=torch.cat([torch.from_numpy(np.concatenate(grids,axis=-1)).permute(0,3,1,2),tags,
                    static[None,:3].expand(stop,-1,-1,-1)],dim=1)
    ext=torch.cat([static[None,3:].expand(stop,-1,-1,-1),physics],dim=1)
    return core.contiguous(),ext.contiguous()


def statistics(x,y,idx):
    # Train-only feature scaling. Use the same noncentered scale for both target components.
    a=x[idx]; mean=a.mean((0,2,3),keepdim=True)
    sd=a.std((0,2,3),unbiased=False,keepdim=True).clamp_min(.04)
    return mean,sd,y[idx].square().mean().sqrt()


def train(x,y,tr,va,spec,seed,epochs,device,inner):
    head,_,arch,aux,extreme=spec; mean,sd,scale=statistics(x,y,tr)
    mean,sd,scale=mean.to(device),sd.to(device),scale.to(device)
    torch.random.default_generator.manual_seed(seed)
    model=WindNet(x.shape[1],24,head,arch).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.001)
    rng=np.random.default_rng(seed)
    # First-order approximation of normalized RMSE evaluated at a fixed train-only error scale.
    rm=y[tr].std((0,2,3),unbiased=False).clamp_min(.1)
    weights=(torch.tensor(OFFICIAL_SCALE[2:],dtype=torch.float32)*rm).reciprocal()
    weights=(weights/weights.sum()).to(device)
    hist=[]; counts=[]; best=float('inf'); champion=None
    indices=tr
    for epoch in range(epochs):
        model.train()
        for j in np.array_split(rng.permutation(indices),max(1,(len(indices)+23)//24)):
            a=(x[j].to(device)-mean)/sd; b=y[j].to(device)/scale
            opt.zero_grad(set_to_none=True); obj=loss(model(a),b,weights,aux,extreme)
            obj.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2.); opt.step()
        if inner:
            p=infer(model,x,va,mean,sd,scale,device)
            metric=float((np.sqrt(np.mean((p-y[va].numpy())**2,axis=(0,2,3)))/OFFICIAL_SCALE[2:]).sum())
            hist.append(metric)
            if metric<best: best=metric; champion=epoch+1
        counts.append(len(indices))
    if inner: return champion,hist
    pred=infer(model,x,va,mean,sd,scale,device).transpose(0,2,3,1)
    return pred,dict(model={k:v.cpu() for k,v in model.state_dict().items()},input_mean=mean.cpu(),
                     input_sd=sd.cpu(),wind_scale=scale.cpu(),width=24,epochs=epochs,spec=list(spec),seed=seed)


@torch.inference_mode()
def infer(model,x,ix,mean,sd,scale,device):
    model.eval()
    return np.concatenate([(model((x[j].to(device)-mean)/sd)[0]*scale).cpu().numpy()
                           for j in np.array_split(ix,max(1,(len(ix)+23)//24))])


def diagnostics(pred,truth,base,static,threshold,t):
    out={}; p,b,y=pred[...,2:],base[...,2:],truth[...,2:]
    s=np.linalg.norm(y,axis=-1); lat=static['wrf_lats']; lon=static['wrf_lons']; dem=static['dem']
    masks={'all':np.ones_like(s,dtype=bool),'strong':s>=threshold,'calm':s<.5,
           'high':np.broadcast_to(dem>=np.quantile(dem,.75),s.shape),
           'low':np.broadcast_to(dem<=np.quantile(dem,.25),s.shape),
           'south':np.broadcast_to(lat[:,None]<=np.quantile(lat,1/3),s.shape),
           'southwest':np.broadcast_to((lat[:,None]<=np.quantile(lat,1/3))&
                                      (lon[None]<=np.quantile(lon,1/3)),s.shape)}
    for name,m in masks.items():
        if not m.any(): continue
        def metrics(a):
            wind=a[m]; target=y[m]; aw=np.linalg.norm(wind,axis=-1); yw=s[m]; valid=(yw>=.5)&(aw>=.05)
            dot=(wind[valid]*target[valid]).sum(-1)/(aw[valid]*yw[valid]); angle=np.rad2deg(np.arccos(dot.clip(-1,1)))
            return dict(uv=np.sqrt(np.mean((wind-target)**2,axis=0)).tolist(),speed_bias=float((aw-yw).mean()),
                        direction_mae=float(angle.mean()) if len(angle) else None)
        out[name]=dict(count=int(m.sum()),candidate=metrics(p),reference=metrics(b))
    return out


def entry(d,cut,pred,base):
    tr,va,_,t,(start,_),_,_,clim=d.fold(cut); ds=d.ds
    truth=ds.y[va]; m=score_mask(t); dates=(t.normalize()-pd.Timestamp(start,tz='UTC')).days
    blocks={}
    for k in (3,7):
        blocks[str(k)]=[float(evaluate(pred[dates//k==i],truth[dates//k==i],t[dates//k==i],clim[dates//k==i])['official']-
                         evaluate(base[dates//k==i],truth[dates//k==i],t[dates//k==i],clim[dates//k==i])['official'])
                         for i in np.unique(dates//k) if (m&(dates//k==i)).any()]
    threshold=float(np.quantile(np.linalg.norm(ds.y[tr[::3],...,2:],axis=-1),.9))
    return dict(metrics=evaluate(pred,truth,t,clim),reference=evaluate(base,truth,t,clim),blocks=blocks,
                extreme_threshold=threshold,groups=diagnostics(pred[m],truth[m],base[m],ds.static,threshold,t[m]))


def verify(d,cut,summary,base):
    for key in ('prediction','checkpoint'):
        link=summary[key]; assert sha(ROOT/link['path'])==link['sha256']
    meta,(p,)=aligned_predictions([ROOT/summary['prediction']['path']]); assert_reference(meta,d,cut)
    np.testing.assert_array_equal(p[...,:2],base[...,:2]); assert entry(d,cut,p,base)==summary['evaluation']
    torch.load(ROOT/summary['checkpoint']['path'],weights_only=True,map_location='cpu')


def ranking(state):
    rows=[]
    for name,result in state['experiments'].items():
        entries=result['cuts']
        delta=[v['evaluation']['metrics']['official']-v['evaluation']['reference']['official'] for v in entries.values()]
        rows.append((name,result['pooled']['official'],min(delta),delta))
    return sorted(rows,key=lambda r:r[1],reverse=True)


def run(args):
    torch.set_num_threads(3); proto,digest=protocol()
    rs=ROOT/'research_state.json'; r5=read(R5/'research_state.json'); d=Development()
    for cut in proto['development']: assert d.bounds[cut]==proto['folds'][cut]
    state=read(rs) if rs.exists() else dict(started=time.time(),protocol_sha256=digest,experiments={},
         reference='R5/ensemble-05-1',reference_state_sha256=sha(R5/'research_state.json'))
    assert state['protocol_sha256']==digest and state['reference_state_sha256']==sha(R5/'research_state.json')
    if args.audit:
        for v in state['experiments'].values():
            for cut,s in v['cuts'].items(): verify(d,cut,s,reference(d,cut,r5))
            assert v['pooled']==pool([s['evaluation']['metrics'] for s in v['cuts'].values()])
        print('Verified:',ranking(state),flush=True); return
    write(rs,state)
    from functools import lru_cache
    @lru_cache(maxsize=1)
    def tensors(stop): return input_fields(d.ds,stop)
    for model in args.models:
        name=f'{model}-s{args.seed}-e{args.epochs}'
        if name in state['experiments']: continue
        if len(state['experiments'])>=min(60,proto['budget']['max_configurations']): break
        start=time.time(); cuts={}; spec=SPECS[model]
        for cut in proto['development']:
            if time.time()-state['started']>min(180,proto['budget']['minutes'])*60: return
            directory=ROOT/'runs'/name/cut; summary=directory/'summary.json'; base=reference(d,cut,r5)
            if summary.exists():
                cuts[cut]=read(summary); verify(d,cut,cuts[cut],base); continue
            tr,va,_,t,bounds,*_=d.fold(cut)
            inner_end=pd.Timestamp(bounds[0],tz='UTC'); inner_start=inner_end-pd.Timedelta(days=7)
            times=d.ds.times; it=tr[times[tr]<inner_start]; iv=tr[(times[tr]>=inner_start)&score_mask(times[tr])]
            assert len(iv) and times[it].max()<times[iv].min() and times[tr].max()<times[va].min()
            core,ext=tensors(len(d.ds.times[times<pd.Timestamp(proto['folds']['C2']['end_exclusive'],tz='UTC')]))
            # Fixed feature shape and parameter count across the physics ablation.
            x=torch.cat([core,ext if spec[1] else torch.zeros_like(ext)],dim=1)
            y=torch.from_numpy(d.ds.y[:len(x),...,2:].copy()).permute(0,3,1,2).contiguous()
            with (lease(proto) if args.device=='cuda' else nullcontext()):
                device=torch.device(args.device)
                epoch,hist=train(x,y,it,iv,spec,args.seed,args.epochs,device,True)
                wind,fit=train(x,y,tr,va,spec,args.seed,epoch,device,False)
            full=base.copy(); full[...,2:]=wind
            if directory.exists(): directory.rename(directory.with_name(f'{cut}.partial-{time.time_ns()}'))
            out=Experiment(directory); path=directory/'best.pt'; torch.save(fit,path)
            out.predictions('predictions',full,t,d.ds.static,score_mask(t),*bounds)
            def link(p): return dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p))
            result=dict(evaluation=entry(d,cut,full,base),epoch=epoch,inner_curve=hist,inner_start=inner_start.isoformat(),
                        prediction=link(directory/'predictions.npz'),checkpoint=link(path),elapsed=time.time()-start)
            verify(d,cut,result,base); out.json('summary',result); cuts[cut]=result
            print(name,cut,epoch,round(result['evaluation']['metrics']['official'],6),flush=True)
        state['experiments'][name]=dict(architecture=list(spec),seed=args.seed,max_epochs=args.epochs,cuts=cuts,
              pooled=pool([v['evaluation']['metrics'] for v in cuts.values()]),seconds=time.time()-start)
        write(rs,state)
        print('Rank:',ranking(state)[:3],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('models',choices=tuple(SPECS),nargs='*'); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--epochs',type=int,default=8); p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--audit',action='store_true'); run(p.parse_args())
