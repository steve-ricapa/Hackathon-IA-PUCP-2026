"""Channel-specific ensembles selected and calibrated only on earlier folds."""
from functools import lru_cache

import numpy as np
import torch

from train.agent_b.round4.run import DEV, pool, read, sha
from train.shared.artifacts import aligned_predictions
from train.shared.metrics import score_mask

from .loop import ROOT


@lru_cache(maxsize=32)
def artifact(name,cut,path,expected):
    path=ROOT/path
    assert sha(path)==expected
    meta,(a,)=aligned_predictions([path])
    return meta,a


def prediction(data,pool_,name,cut):
    src=pool_[name]['cuts'][cut]['prediction']
    meta,a=artifact(name,cut,src['path'],src['sha256'])
    np.testing.assert_array_equal(meta['times'],data.fold(cut)[3].strftime('%Y%m%d%H'))
    return a


def fit(data,cut,spec):
    old=DEV[:DEV.index(cut)]; base=data.incumbent(cut); output=base.copy()
    state=read(ROOT/'research_state.json'); champion=state['incumbent']
    if 'fixed' in spec:
        # Follow-up to the observed scalar/wind trade-off; development-only evidence.
        name=spec['fixed']; output[...,2:]=prediction(data,state['experiments'],name,cut)[...,2:]
        return output,dict(source=state['experiments'][name]['cuts'][cut]['prediction'],channels=[2,3])
    if not old: return output,dict(identity=True)
    available={n:x for n,x in state['experiments'].items() if x['family'] in spec['families']}
    group=spec['group']
    info=[data.fold(c) for c in old]; masks=[score_mask(m[3]) for m in info]
    ix=np.concatenate([m[1][mask] for m,mask in zip(info,masks)])
    bp=np.concatenate([data.incumbent(c)[mask] for c,mask in zip(old,masks)])
    coefs=[]; names=[]
    for j in range(4):
        if group=='wind' and j<2 or group=='scalars' and j>=2:
            coefs.append(torch.empty(0)); names.append([]); continue
        def value(entry):
            return pool([entry['cuts'][c]['score'] for c in old])['rmse'][j]
        base_rmse=value(champion)
        chosen=[]; seen=set()
        for name in sorted(available,key=lambda n:value(available[n])):
            # Do not select two blend weights of the same fit.
            kind=name.rsplit('-',1)[0]
            if kind in seen or value(available[name])>=base_rmse: continue
            if spec['consistent'] and any(available[name]['cuts'][c]['score']['rmse'][j]>
                                          champion['cuts'][c]['score']['rmse'][j] for c in old): continue
            chosen.append(name); seen.add(kind)
            if len(chosen)==spec['size']: break
        if not chosen:
            coefs.append(torch.empty(0)); names.append([]); continue
        a=[]; v=[]
        for name in chosen:
            a.append((np.concatenate([prediction(data,available,name,c)[mask,...,j] for c,mask in zip(old,masks)])-bp[...,j]).ravel())
            v.append(prediction(data,available,name,cut)[...,j]-base[...,j])
        if spec['ridge'] is None:
            w=torch.ones(len(chosen))/len(chosen)
        else:
            x=torch.from_numpy(np.stack(a,axis=1)); y=torch.from_numpy((data.ds.y[ix,...,j]-bp[...,j]).ravel())
            gr=x.T@x; gr.diagonal().add_(spec['ridge']*gr.diagonal().mean().clamp_min(1))
            w=torch.cholesky_solve((x.T@y)[:,None],torch.linalg.cholesky(gr)).flatten().clamp_min(0)
            w=w/w.sum().clamp_min(1.)
        output[...,j]+=np.einsum('i,itxy->txy',w.numpy(),np.stack(v))
        coefs.append(w); names.append(chosen)
    return output,dict(coef=coefs,models=names,earlier=list(old),sources={name:{c:available[name]['cuts'][c]['prediction']
                     for c in (*old,cut)} for name in set(sum(names,[]))})


WEIGHTS=(.5,1.)


def configurations():
    all_models=('spectral','regime','regional')
    settings=[(all_models,'all',1,False,None),(all_models,'all',3,True,None),
              (all_models,'all',3,False,.25),(all_models,'wind',3,True,.25)]
    for i,(families,group,size,consistent,ridge) in enumerate(settings,1):
        yield f'ensemble-{i:02}',dict(families=families,group=group,size=size,consistent=consistent,ridge=ridge)
    yield 'ensemble-05',dict(fixed='spectral-06-0.2')
    yield 'ensemble-06',dict(fixed='spectral-07-0.2')
