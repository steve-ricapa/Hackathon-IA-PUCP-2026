"""Nested selectors and artifact audit for six CPU fits and two blend diagnostics."""
import argparse
import time

import numpy as np
import pandas as pd
import psutil
import torch

from train.agent_b.round4.run import ORIG, ROOT as R4, frozen_config, pool, read, sha, write
from train.agent_b.round5.loop import Development
from train.agent_b.round6.run import assert_reference
from train.shared.artifacts import Experiment, aligned_predictions
from train.shared.metrics import evaluate, score_mask
from .models import krr, mos, mos_fields


HERE=ORIG/'.private/experiments'
ROOT=HERE/'round7/agent_b'
DEV=('C0','C1','C2')
SPECS={
    'linear':('linear',(0,),True), 'rbf':('rbf',(0,),True), 'clock':('clock',(0,),True),
    'clock-lags':('clock',(2,1,0),True), 'mos-linear':'linear', 'mos-nonlinear':'nonlinear',
    'regional-local':None, 'channel-oof':None,
}


def protocol():
    root=HERE/'round7' if (HERE/'round7/READY').is_file() else HERE/'round6'
    path=root/'protocol.json'; digest=sha(path); p=read(path)
    assert digest==(root/'READY').read_text().strip() and tuple(p['development'])==DEV
    return p,digest,root.name


def index(path): return {'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path)}


def assess(d,cut,p,b):
    _,va,_,t,(start,_),_,_,c=d.fold(cut); y=d.ds.y[va]
    day=np.asarray((t.normalize()-pd.Timestamp(start,tz='UTC')).days)
    blocks={}
    for k in (3,7):
        out=[]
        for j in np.unique(day//k):
            m=day//k==j
            if not score_mask(t[m]).any(): continue
            a=evaluate(p[m],y[m],t[m],c[m]); r=evaluate(b[m],y[m],t[m],c[m])
            out.append(dict(hours=a['hours'],delta=a['official']-r['official'],rmse_delta=(np.array(a['rmse'])-r['rmse']).tolist()))
        blocks[str(k)]=out
    return dict(metrics=evaluate(p,y,t,c),blocks=blocks)


def load(d,cut,e):
    p=ROOT/e['prediction']['path']; assert sha(p)==e['prediction']['sha256']
    meta,(arr,)=aligned_predictions([p]); assert_reference(meta,d,cut)
    return arr


def audit(d,cut,e,base):
    p=load(d,cut,e); assert assess(d,cut,p,base)==e['evaluation']
    ck=ROOT/e['checkpoint']['path']; assert sha(ck)==e['checkpoint']['sha256']
    torch.load(ck,map_location='cpu',weights_only=True)


def fit(d,cut,name,fields):
    ds=d.ds; tr,va,_,_,(start,_),_,_,_=d.fold(cut)
    pivot=pd.Timestamp(start,tz='UTC')-pd.Timedelta(days=7); it=tr[ds.times[tr]<pivot]; iv=tr[ds.times[tr]>=pivot]
    assert ds.times[it[-1]]<ds.times[iv[0]]<=ds.times[iv[-1]]<ds.times[va[0]]
    if name.startswith('mos'):
        def model(a,b,params): return mos(ds,a,b,SPECS[name],params,fields)
        grid=(.025,.15)
    else:
        def model(a,b,params): return krr(ds,a,b,SPECS[name],params)
        grid=tuple((w,r) for w in ((1.,) if name=='linear' else (.45,1.6)) for r in (.05,1.0))
    scores=[(k,evaluate(y,ds.y[iv],ds.times[iv],np.zeros_like(y))['official']) for k,y,_ in model(it,iv,grid)]
    choice=max(scores,key=lambda v:v[1])[0]; _,pred,ck=next(model(tr,va,(choice,)))
    return pred,ck,dict(inner_start=pivot.isoformat(),parameters=choice,inner_scores=scores)


def combine(d,state,cut,name,base):
    runs=state['experiments']; earlier=DEV[:DEV.index(cut)]
    def source(k): return load(d,cut,runs[k]['cuts'][cut])
    if name=='regional-local':
        keys=['clock','mos-nonlinear']
        return .5*source(keys[0])+.5*source(keys[1]),dict(sources=keys,weight=.5),dict(rule='fixed equal-weight models')
    keys=list(SPECS)[:-2]; selected=[]; out=base.copy(); cache={}
    for j in range(4):
        def error(k):
            e=state['reference'] if k=='reference' else {c:runs[k]['cuts'][c]['evaluation']['metrics'] for c in DEV}
            return pool([e[c] for c in earlier])['rmse'][j]
        k=min(['reference',*keys],key=error) if earlier else 'reference'; selected.append(k)
        if k=='reference': continue
        if k not in cache: cache[k]=source(k)
        out[...,j]=base[...,j]+.25*(cache[k][...,j]-base[...,j])
    return out,dict(sources=selected,weight=.25),dict(earlier=earlier,channels=selected)


def close(d,state,bases):
    if not state['experiments']: raise RuntimeError('No completed experiments')
    if set(state['experiments'])!=set(SPECS) and time.time()-state['started']<45*60:
        raise RuntimeError('Complete the planned designs before freezing')
    comparator=pool(list(state['reference'].values())); shortlist=[]; diag={}; blends={}
    for name,e in state['experiments'].items():
        for c in DEV: audit(d,c,e['cuts'][c],bases[c])
        scores=[e['cuts'][c]['evaluation']['metrics'] for c in DEV]
        weeks=[b['delta'] for c in DEV for b in e['cuts'][c]['evaluation']['blocks']['7']]
        stable=(all(s['official']>=state['reference'][c]['official']-1e-9 for s,c in zip(scores,DEV))
                and sum(x>1e-9 for x in weeks)>len(weeks)/2)
        if stable and e['pooled']['official']>comparator['official']: shortlist.append(name)
        diag[name]=[float(x-y) for x,y in zip(e['pooled']['rmse'],comparator['rmse'])]
        # Mixing all channels at once computes exactly the same channel RMSEs
        # as four isolated single-channel blends. Do not select on these scans.
        if SPECS[name] is not None:
            for c in DEV:
                p=bases[c]+.25*(load(d,c,e['cuts'][c])-bases[c])
                _,va,_,t,_,_,_,clim=d.fold(c)
                blends.setdefault(name,{})[c]=evaluate(p,d.ds.y[va],t,clim)
    best=max(state['experiments'],key=lambda k:state['experiments'][k]['pooled']['official'])
    promoted=max(shortlist,key=lambda k:state['experiments'][k]['pooled']['official']) if shortlist else 'reference'
    candidates={}
    for n in dict.fromkeys([best,*shortlist]):
        e=state['experiments'][n]; candidates[n]=dict(predictions={c:v['prediction']['path'] for c,v in e['cuts'].items()},
          metadata=dict(configuration=e['spec'],pooled=e['pooled'],artifacts=e['cuts'],integration_recommended=n in shortlist))
    # A's R7 intake asserts bit-identical RH/T for all B entries; keep the
    # unmasked models in diagnostics rather than handing A incompatible arrays.
    wind=min(state['experiments'],key=lambda k:sum(np.array(state['experiments'][k]['pooled']['rmse'][2:])/[2.6776,2.0876]))
    wind_metrics={}; wind_links={}
    for c in DEV:
        b=bases[c]; p=b.copy(); p[...,2:]=load(d,c,state['experiments'][wind]['cuts'][c])[...,2:]
        _,_,_,t,bounds,*_=d.fold(c)
        dest=ROOT/'intake'/wind/c
        if not dest.exists(): Experiment(dest).predictions('predictions',p,t,d.ds.static,score_mask(t),*bounds)
        wind_links[c]=index(dest/'predictions.npz'); saved=load(d,c,{'prediction':wind_links[c]})
        np.testing.assert_array_equal(saved,p); wind_metrics[c]=assess(d,c,saved,b)
    wind_pooled=pool([v['metrics'] for v in wind_metrics.values()])
    models={wind+'-wind':dict(predictions={c:v['path'] for c,v in wind_links.items()},metadata=dict(
        wind_only=True,configuration=state['experiments'][wind]['spec'],source=wind,pooled=wind_pooled,
        evaluation=wind_metrics,artifacts=wind_links,integration_recommended=False))}
    deltas={k:(np.array(pool(list(v.values()))['rmse'])-comparator['rmse']).tolist() for k,v in blends.items()}
    selected=dict(promoted=promoted,best_screened=best,comparator=comparator,diagnostic_channel_rmse_deltas=diag,
                  diagnostic_channel_rmse_deltas_at_25pct=deltas)
    write(ROOT/'best_config.json',selected)
    state['closed']=True; state['actual_seconds']=round(time.time()-state['started'],2)
    write(ROOT/'research_state.json',state)
    write(ROOT/'DONE.json',dict(protocol_sha256=state['protocol_sha256'],protocol_round=state['protocol_round'],
         status='development_only',c3='not accessed',selection=selected,models=models,full_field_diagnostics=candidates,
         fitted_configurations=sum(SPECS[k] is not None for k in state['experiments']),
         blend_diagnostics=sum(SPECS[k] is None for k in state['experiments']),
         research_state_sha256=sha(ROOT/'research_state.json')))
    print('Selection:',selected,flush=True)


def run(args):
    torch.set_num_threads(2); proto,digest,origin=protocol(); d=Development(); frozen_config()
    old=read(R4/'research_state.json'); d.previous=old['experiments']['oof-0.25']['cuts']
    bases={c:d.incumbent(c) for c in DEV}
    for c in DEV: assert d.bounds[c]==proto['folds'][c]
    rs=ROOT/'research_state.json'; state=read(rs) if rs.exists() else dict(started=time.time(),protocol_sha256=digest,
       protocol_round=origin,reference_hash=sha(R4/'best_config.json'),reference={c:assess(d,c,bases[c],bases[c])['metrics'] for c in DEV},experiments={})
    assert state['protocol_sha256']==digest and state['reference_hash']==sha(R4/'best_config.json')
    if args.close:
        close(d,state,bases); return
    if state.get('closed'): raise RuntimeError('Round already frozen')
    write(rs,state); fields=None
    for name in args.models:
        if name not in SPECS: raise ValueError(name)
        if name in state['experiments']: continue
        cuts={}; started=time.perf_counter()
        for c in DEV:
            if time.time()-state['started']>45*60: return
            directory=ROOT/'runs'/name/c; summary=directory/'summary.json'
            if summary.exists(): cuts[c]=read(summary); audit(d,c,cuts[c],bases[c]); continue
            if name.startswith('mos') and fields is None:
                fields=mos_fields(d.ds,d.fold('C2')[1][-1]+1)
            elif not name.startswith('mos'):
                fields=None
            if name in list(SPECS)[-2:]: pred,ck,details=combine(d,state,c,name,bases[c])
            else: pred,ck,details=fit(d,c,name,fields)
            if directory.exists(): directory.rename(directory.with_name(f'{c}.partial-{time.time_ns()}'))
            _,_,_,t,bounds,*_=d.fold(c); out=Experiment(directory); checkpoint=directory/'best.pt'; torch.save(ck,checkpoint)
            out.predictions('predictions',pred,t,d.ds.static,score_mask(t),*bounds)
            ram=psutil.Process().memory_info()
            e=dict(selection=details,evaluation=assess(d,c,pred,bases[c]),prediction=index(directory/'predictions.npz'),
                   checkpoint=index(checkpoint),peak_mib=getattr(ram,'peak_wset',ram.rss)/2**20)
            audit(d,c,e,bases[c]); out.json('summary',e); cuts[c]=e
        result=pool([v['evaluation']['metrics'] for v in cuts.values()]); state['experiments'][name]=dict(
           spec=SPECS[name],pooled=result,cuts=cuts,seconds=time.perf_counter()-started)
        write(rs,state); print(name,result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('models',nargs='*'); p.add_argument('--close',action='store_true'); run(p.parse_args())
