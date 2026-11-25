"""Chronological regularized stacking with smooth weather-regime gates."""
from functools import lru_cache

import numpy as np
import torch

from train.agent_b.round4.run import DEV, R3, read
from train.shared.artifacts import aligned_predictions
from train.shared.metrics import score_mask


@lru_cache(maxsize=1)
def member_paths():
    manifests={t:read(R3/f'agent_{t.lower()}'/'DEVELOPMENT.json')['models'] for t in 'AB'}
    paths={c:[] for c in DEV}
    for key in read(R3/'integration/frozen.json')['weights']:
        t,n=key.split('/',1)
        files=manifests[t][n]['predictions']
        if all(c in files for c in DEV):
            for c in DEV: paths[c].append(R3/f'agent_{t.lower()}'/files[c])
    return paths


@lru_cache(maxsize=3)
def displacement(data,cut):
    p=data.incumbent(cut)
    meta,values=aligned_predictions(member_paths()[cut])
    np.testing.assert_array_equal(meta['times'],data.fold(cut)[3].strftime('%Y%m%d%H'))
    return torch.from_numpy(np.stack([x-p for x in values],axis=-1))


def context(data,cut,kind):
    _,ix,h,*_=data.fold(cut)
    x=data.ds.x[ix]; u,v=x[...,2],x[...,3]
    if kind=='speed': return np.hypot(u,v).mean((1,2))[:,None]
    if kind=='v': return v.mean((1,2))[:,None]
    if kind=='moist': return x[...,0].mean((1,2))[:,None]
    if kind=='vector': return x[...,2:].mean((1,2))
    if kind=='diurnal': return np.cos(h[ix]*np.pi/12)[:,None]
    return np.zeros((len(ix),0),dtype=np.float32)


def regions(static,method):
    if method=='relief': return (static['dem']>=np.median(static['dem'])).astype('int64')
    if method=='latitude':
        y=static['wrf_lats'];return np.broadcast_to((y[:,None]>=np.median(y)).astype('int64'),static['dem'].shape)
    return np.zeros_like(static['dem'],dtype='int64')


def basis(g,coef):
    center,scale=coef
    v=torch.from_numpy(np.tanh((g-center)/scale).astype('float32'))
    return torch.cat([torch.ones((len(v),1)),v],dim=1)


def fit(data,cut,spec):
    old=DEV[:DEV.index(cut)]; base=data.incumbent(cut)
    if not old: return base.copy(),dict(identity=True)
    d=displacement(data,cut); mode=spec['gate']; reg=regions(data.ds.static,spec['regions'])
    info=[data.fold(c) for c in old]; masks=[score_mask(m[3]) for m in info]
    ix=np.concatenate([m[1][mask] for m,mask in zip(info,masks)])
    ctx=np.concatenate([context(data,c,mode)[mask] for c,mask in zip(old,masks)])
    center=ctx.mean(0); scale=ctx.std(0).clip(1e-4); g=basis(ctx,(center,scale))
    gv=basis(context(data,cut,mode),(center,scale))
    delta=torch.cat([displacement(data,c)[mask] for c,mask in zip(old,masks)])
    y=torch.from_numpy(data.ds.y[ix]-np.concatenate([data.incumbent(c)[mask] for c,mask in zip(old,masks)]))
    output=torch.from_numpy(base.copy()); coeffs=[]
    for k in np.unique(reg):
        p=reg==k
        a=delta[:,p].permute(2,0,1,3)
        x=(a[...,None]*g[None,:,None,None,:]).reshape(4,-1,a.shape[-1]*g.shape[-1])
        target=y[:,p].permute(2,0,1).reshape(4,-1,1)
        gram=x.transpose(1,2)@x
        magnitude=gram.diagonal(dim1=-2,dim2=-1)[:,:g.shape[-1]].mean(-1).clamp_min(1)
        gram.diagonal(dim1=-2,dim2=-1).add_(spec['ridge']*magnitude[:,None])
        w=torch.cholesky_solve(x.transpose(1,2)@target,torch.linalg.cholesky(gram)).reshape(4,a.shape[-1],g.shape[-1])
        w=w/w.abs().sum(dim=(1,2),keepdim=True).div(1.5).clamp_min(1.)
        output[:,p]+=torch.einsum('tpcm,cmg,tg->tpc',d[:,p],w,gv)
        coeffs.append(w)
    return output.numpy(),dict(coef=torch.stack(coeffs),center=torch.from_numpy(center),scale=torch.from_numpy(scale),
                               regions=torch.from_numpy(reg.copy()),members=[p.name for p in member_paths()[cut]])


def configurations():
    settings=[('none','all',.1),('speed','all',.15),('v','all',.15),('moist','all',.15),
              ('vector','all',.25),('diurnal','all',.15),('vector','latitude',.3),('speed','relief',.3)]
    for i,(gate,region,ridge) in enumerate(settings,1):
        yield f'regime-{i:02}',dict(gate=gate,regions=region,ridge=ridge)
