"""Causal dynamics benchmark with nested training-only model selection."""
import itertools

import numpy as np
import pandas as pd
import torch

from train.agent_b.round4.run import pool, read, sha, write
from train.agent_b.round5.loop import Development, ROOT as R5
from train.agent_b.round5.spectral import dct
from train.shared.artifacts import Experiment, aligned_predictions
from train.shared.metrics import evaluate, score_mask
from train.shared.preprocess import temporal_windows
from .run import ROOT, assert_reference, protocol, reference


ROOT=ROOT/'high_impact'/'temporal'
DESIGNS=list(itertools.product(('absolute','anomaly','tendency'),((2,1,0),(6,3,0),(12,9,6,3,0))))
DESIGNS.extend((mode,(2,1,0)) for mode in ('absolute-kernel','anomaly-kernel','tendency-kernel'))
DESIGNS.extend((mode,(6,3,0)) for mode in ('transport-zero','transport'))
PENALTIES=(.02,.10,.40)


def transport(w,m,lags,static,advect):
    """Transport historic coarse-grid anomalies with a frozen local flow proxy."""
    from torch.nn.functional import grid_sample
    lat=np.asarray(static['gfs_lats']); lon=np.asarray(static['gfs_lons'])
    if not np.allclose(np.diff(lat),np.diff(lat)[0],atol=2e-5) or not np.allclose(np.diff(lon),np.diff(lon)[0],atol=2e-5):
        raise ValueError('Expected regular coarse-grid axes')
    dy=np.float32(np.deg2rad(np.diff(lat).mean())*6371000)
    dx=(np.deg2rad(np.diff(lon).mean())*6371000*np.cos(np.deg2rad(lat))).astype('float32')[:,None]
    r,c=np.meshgrid(np.arange(len(lat),dtype='float32'),np.arange(len(lon),dtype='float32'),indexing='ij')
    out=w.copy(); available=~m.copy()
    for j,lag in enumerate(lags[:-1]):
        # Backtrace with the historical GFS vector, not the diurnal anomaly.
        past=w[:,j]; wind=advect[:,j,...,2:]
        x=c-wind[...,0]*(3600*lag)/dx; y=r-wind[...,1]*(3600*lag)/dy
        inside=(x>=0)&(x<=len(lon)-1)&(y>=0)&(y<=len(lat)-1)&~m[:,j,None,None]
        grid=torch.from_numpy(np.stack([2*x/(len(lon)-1)-1,2*y/(len(lat)-1)-1],axis=-1).astype('float32'))
        carried=grid_sample(torch.from_numpy(past).permute(0,3,1,2),grid,align_corners=True).permute(0,2,3,1).numpy()
        out[:,j]=np.where(inside[...,None],carried,past)
        available[:,j]=inside.reshape(len(w),-1).any(1)
    return out,~available


def encode(x,t,tr,mode,lags,static=None):
    nonlinear=mode.endswith('-kernel'); mode=mode.removesuffix('-kernel')
    hours=t.hour.to_numpy()
    climate=np.stack([x[tr[hours[tr]==h]].mean(0) for h in range(24)])
    assert np.isfinite(climate).all()
    source=x if mode=='absolute' else x-climate[hours]
    w,m=temporal_windows(source,t,lags)
    if mode=='transport':
        physical,_=temporal_windows(x,t,lags)
        w,m=transport(w,m,lags,static,physical)
    if mode in ('tendency','transport','transport-zero'):
        delta=np.diff(w,axis=1)/(-np.diff(lags))[None,:,None,None,None]
        delta=np.where((m[:,1:]|m[:,:-1])[:,:,None,None,None],0,delta)
        w=np.concatenate([delta,w[:,-1:]],axis=1).astype('float32')
    g=dct(w,7,7).reshape(len(x),-1)
    angle=hours*np.pi/12
    a=torch.as_tensor(np.column_stack([g,m,np.sin(angle),np.cos(angle)]),dtype=torch.float32)
    train=a[tr]; mu=train.mean(0); sd=train.std(0,unbiased=False).clamp_min(.025)
    a=(a-mu)/sd; state=dict(gfs_climate=torch.from_numpy(climate),mean=mu,sd=sd)
    if nonlinear:
        # Deterministic, train-only evenly spaced anchor days; fixed capacity.
        anchors=a[tr[np.linspace(0,len(tr)-1,96,dtype=int)]]
        distances=torch.cdist(a,anchors).square()/a.shape[-1]
        h=torch.exp(-distances/2)
        state.update(anchors=anchors,kernel_mean=h[tr].mean(0),kernel_scale=h[tr].std(0,unbiased=False).clamp_min(.05))
        a=torch.cat([a,(h-state['kernel_mean'])/state['kernel_scale']],dim=1)
    return a,state


def fit(ds,tr,va,mode,lags,penalties):
    assert ds.times[tr[-1]]<ds.times[va[0]]
    stop=va[-1]+1; t=ds.times[:stop]; h=t.hour.to_numpy(); y=ds.y
    a,fit=encode(ds.x[:stop],t,tr,mode,lags,ds.static); at,av=a[tr],a[va]
    clim=np.stack([y[tr[h[tr]==k]].mean(0) for k in range(24)])
    target=y[tr]-clim[h[tr]]; z=torch.from_numpy(dct(target,24,16).reshape(len(tr),-1))
    intercept=z.mean(0); gram=at.T@at; rhs=at.T@(z-intercept)
    for p in penalties:
        gr=gram.clone(); gr.diagonal().add_(p*len(tr)); coef=torch.cholesky_solve(rhs,torch.linalg.cholesky(gr))
        pred=dct((av@coef+intercept).numpy().reshape(len(va),24,16,4),24,16,inverse=True,shape=y.shape[1:3])+clim[h[va]]
        yield p,pred,dict(**fit,coef=coef,intercept=intercept,wrf_climate=torch.from_numpy(clim),mode=mode,lags=lags,penalty=p)


def audit_artifact(d,cut,entry):
    for key in ('prediction','checkpoint'):
        art=entry[key]; assert sha(ROOT/art['path'])==art['sha256']
    meta,(p,)=aligned_predictions([ROOT/entry['prediction']['path']]); assert_reference(meta,d,cut)
    _,va,_,t,_,_,_,clim=d.fold(cut)
    assert evaluate(p,d.ds.y[va],t,clim)==entry['official']
    torch.load(ROOT/entry['checkpoint']['path'],map_location='cpu',weights_only=True)


def run():
    torch.set_num_threads(2); proto,digest=protocol(); dev=proto['development']
    d=Development(); src=read(R5/'research_state.json'); ds=d.ds; path=ROOT/'research_state.json'
    state=read(path) if path.exists() else dict(protocol_sha256=digest,experiments={},
              future_context='blocked: protocol is causal; no initialization/lead metadata in official arrays')
    assert state['protocol_sha256']==digest
    bases={c:reference(d,c,src) for c in dev}
    for mode,lags in DESIGNS:
        name=mode+'-'+str(lags[0]); cuts={}
        if name in state['experiments']:
            for cut,entry in state['experiments'][name]['cuts'].items(): audit_artifact(d,cut,entry)
            continue
        for cut in dev:
            tr,va,_,t,(start,end),_,_,c=d.fold(cut); assert dict(start=start,end_exclusive=end)==proto['folds'][cut]
            directory=ROOT/'runs'/name/cut; summary=directory/'summary.json'
            if summary.exists(): cuts[cut]=read(summary); audit_artifact(d,cut,cuts[cut]); continue
            inner=pd.Timestamp(start,tz='UTC')-pd.Timedelta(days=7)
            it=tr[ds.times[tr]<inner]; iv=tr[ds.times[tr]>=inner]
            scores={str(p):evaluate(y,ds.y[iv],ds.times[iv],np.zeros_like(y))['official'] for p,y,_ in fit(ds,it,iv,mode,lags,PENALTIES)}
            best=float(max(scores,key=scores.get)); _,pred,checkpoint=next(fit(ds,tr,va,mode,lags,(best,)))
            base=bases[cut]; truth=ds.y[va]
            only=base.copy(); only[...,2:]=pred[...,2:]
            if directory.exists():
                import time
                directory.rename(directory.with_name(f'{cut}.partial-{time.time_ns()}'))
            out=Experiment(directory)
            out.predictions('predictions',pred,t,ds.static,score_mask(t),start,end)
            ck=directory/'best.pt'; torch.save(checkpoint,ck)
            def link(p): return dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p))
            e=dict(official=evaluate(pred,truth,t,c),wind_only=evaluate(only,truth,t,c),reference=evaluate(base,truth,t,c),
                   prediction=link(directory/'predictions.npz'),checkpoint=link(ck),inner_scores=scores,best_penalty=best)
            audit_artifact(d,cut,e); out.json('summary',e); cuts[cut]=e
        pooled={k:pool([cuts[c][k] for c in dev]) for k in ('official','wind_only','reference')}
        state['experiments'][name]=dict(design=dict(mode=mode,lags=lags),cuts=cuts,pooled=pooled)
        write(path,state)
        print(name,[(k,round(v['official'],8)) for k,v in pooled.items()],flush=True)
    for metric in ('official','wind_only'):
        top=max(state['experiments'],key=lambda name:state['experiments'][name]['pooled'][metric]['official'])
        if metric=='official': result={}
        result[metric]=dict(name=top,**state['experiments'][top])
    write(ROOT/'best_config.json',result)


if __name__=='__main__': run()
