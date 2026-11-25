"""Resumable benchmark; C3 is intentionally not addressable in this CLI."""
import argparse
import importlib
import time
from functools import lru_cache

import numpy as np
import pandas as pd
import torch

from train.agent_b.round4.run import DEV, ROOT as R4, Data, frozen_config, pool, read, sha, write
from train.shared.artifacts import Experiment, aligned_predictions
from train.shared.metrics import evaluate, score_mask

ROOT=R4.parents[1]/'round5'/'agent_b'
CAP=60
INCUMBENT='oof-0.25'


class Development(Data):
    def fold(self,cut):
        if cut not in DEV:
            raise ValueError(f'Non-development fold is forbidden: {cut}')
        return super().fold(cut)

    @lru_cache(maxsize=3)
    def incumbent(self,cut):
        _,_,_,t,bounds,_,_,_=self.fold(cut)
        src=self.previous[cut]['predictions']
        path=R4/src['path']
        if sha(path)!=src['sha256']:
            raise ValueError(f'Incumbent hash mismatch: {cut}')
        meta,(p,)=aligned_predictions([path])
        np.testing.assert_array_equal(meta['times'],t.strftime('%Y%m%d%H'))
        assert (meta['start'].item(),meta['end_exclusive'].item())==bounds
        return p


def score(data,cut,pred,mask=slice(None)):
    _,va,_,t,_,_,_,c=data.fold(cut)
    return evaluate(pred[mask],data.ds.y[va[mask]],t[mask],c[mask])


def slices(data,cut,pred):
    _,_,_,t,(start,_),_,_,_=data.fold(cut)
    day=np.asarray((t.normalize()-pd.Timestamp(start,tz='UTC')).days)
    return {str(i):score(data,cut,pred,day//i[0]==i[1]) for i in
            [*((2,j) for j in range(7)),(7,0),(7,1)] if score_mask(t[day//i[0]==i[1]]).any()}


def verified(data,cut,entry,full=True):
    _,_,_,t,bounds,_,_,_=data.fold(cut)
    path=ROOT/entry['prediction']['path']
    assert sha(path)==entry['prediction']['sha256']
    meta,(p,)=aligned_predictions([path])
    np.testing.assert_array_equal(meta['times'],t.strftime('%Y%m%d%H'))
    np.testing.assert_array_equal(meta['scored'],score_mask(t))
    np.testing.assert_array_equal(meta['wrf_lats'],data.ds.static['wrf_lats'])
    np.testing.assert_array_equal(meta['wrf_lons'],data.ds.static['wrf_lons'])
    assert (meta['start'].item(),meta['end_exclusive'].item())==bounds
    assert score(data,cut,p)==entry['score']
    if full:
        assert slices(data,cut,p)==entry['slices']
        checkpoint=entry['checkpoint']
        if checkpoint:
            ck=ROOT/checkpoint['path']; assert sha(ck)==checkpoint['sha256']
            torch.load(ck,map_location='cpu',weights_only=True)
    return entry


def index(path):
    return {'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path)}


def save(data,name,cut,pred,checkpoint):
    _,_,_,t,bounds,_,_,_=data.fold(cut)
    directory=ROOT/'runs'/name/cut
    summary=directory/'summary.json'
    if summary.exists():
        return verified(data,cut,read(summary))
    if directory.exists():
        # Preserve incomplete work, rather than overwriting or deleting it.
        directory.rename(directory.with_name(f'{cut}.partial-{time.time_ns()}'))
    run=Experiment(directory)
    run.predictions('predictions',pred,t,data.ds.static,score_mask(t),*bounds)
    entry=dict(score=score(data,cut,pred),slices=slices(data,cut,pred),
               prediction=index(directory/'predictions.npz'),checkpoint=checkpoint)
    verified(data,cut,entry)
    run.json('summary',entry)
    return entry


def ranking(state):
    incumb=state['incumbent']; previous=incumb['cuts']; result=[]
    for name,entry in state['experiments'].items():
        deltas=[entry['cuts'][c]['score']['official']-previous[c]['score']['official'] for c in DEV]
        weeks=[v['official']-previous[c]['slices'][k]['official'] for c in DEV
               for k,v in entry['cuts'][c]['slices'].items() if k.startswith('(7,')]
        result.append((name,entry['pooled']['official'],deltas,weeks))
    eligible=[r for r in result if r[1]>incumb['pooled']['official'] and min(r[2])>=-1e-8
              and sum(x>1e-8 for x in r[3])>=3]
    return max(eligible,key=lambda r:r[1],default=(INCUMBENT,incumb['pooled']['official'],[0]*3,[0]*6))


def nested(state):
    options={INCUMBENT:state['incumbent'],**state['experiments']}
    out={}
    for i,cut in enumerate(DEV[1:],1):
        earlier=DEV[:i]
        allowable={name:e for name,e in options.items() if all(
            e['cuts'][c]['score']['official']+1e-8>=options[INCUMBENT]['cuts'][c]['score']['official']
            for c in earlier)}
        # Select using earlier folds only; ties retain the incumbent.
        name=max(allowable,key=lambda k:pool([allowable[k]['cuts'][c]['score'] for c in earlier])['official'])
        result=options[name]['cuts'][cut]['score']
        out[cut]=dict(selected=name,inner=earlier,score=result,delta=result['official']-options[INCUMBENT]['cuts'][cut]['score']['official'])
    return out


def persist(state):
    name,value,deltas,weeks=ranking(state)
    state['best']=dict(name=name,official=value,fold_deltas=deltas,week_deltas=weeks)
    state['nested_replay']=nested(state)
    write(ROOT/'research_state.json',state)
    source=state['incumbent'] if name==INCUMBENT else state['experiments'][name]
    write(ROOT/'best_config.json',dict(**state['best'],result=source,selection=state['selection'],
                                      archived_incumbent_sha256=state['parent']['freeze_sha256']))


def run(args):
    torch.set_num_threads(3)
    freeze=frozen_config(); prior=read(R4/'research_state.json')
    d=Development(); d.previous=prior['experiments'][INCUMBENT]['cuts']
    f=ROOT/'research_state.json'
    state=read(f) if f.exists() else dict(experiments={},parent=dict(commit='2739718',
      freeze_sha256=sha(R4/'best_config.json'),development_sha256=sha(R4/'research_state.json')),
      started=time.time(),cap=CAP,selection='Higher pooled score, no worse cut, at least three improved calendar weeks.',
      incumbent=dict(pooled=freeze['pooled'],cuts={c:dict(score=score(d,c,d.incumbent(c)),
          slices=slices(d,c,d.incumbent(c))) for c in DEV}))
    assert state['parent']['freeze_sha256']==sha(R4/'best_config.json')
    if args.family=='audit':
        for e in state['experiments'].values():
            for c in DEV: verified(d,c,e['cuts'][c])
            assert e['pooled']==pool([e['cuts'][c]['score'] for c in DEV])
        print(f"Verified {len(state['experiments'])} configurations; best: {ranking(state)}")
        return
    family=importlib.import_module(f'.{args.family}',__package__)
    weights=family.WEIGHTS if hasattr(family,'WEIGHTS') else (.2,.4)
    for key,spec in family.configurations():
        names=[f'{key}-{w:g}' for w in weights]
        if all(n in state['experiments'] for n in names):
            continue
        remaining=[(n,w) for n,w in zip(names,weights) if n not in state['experiments']][:CAP-len(state['experiments'])]
        if not remaining or time.time()-state['started']>=180*60:
            break
        start=time.perf_counter(); entries={n:{} for n,_ in remaining}
        for c in DEV:
            todo=[]
            for n,w in remaining:
                s=ROOT/'runs'/n/c/'summary.json'
                if s.exists(): entries[n][c]=verified(d,c,read(s))
                else: todo.append((n,w))
            if not todo: continue
            p,fit=family.fit(d,c,spec)
            ck=ROOT/'fits'/key/f'{c}.pt'; ck.parent.mkdir(parents=True,exist_ok=True)
            if not ck.exists():
                tmp=ck.with_suffix('.tmp'); torch.save(fit,tmp); tmp.replace(ck)
            incumbent=d.incumbent(c)
            for n,w in todo:
                entries[n][c]=save(d,n,c,incumbent+w*(p-incumbent),index(ck))
        for n,w in remaining:
            state['experiments'][n]=dict(family=args.family,spec=spec,weight=w,cuts=entries[n],
                pooled=pool([entries[n][c]['score'] for c in DEV]),shared_seconds=time.perf_counter()-start)
            persist(state)
            print(n,round(state['experiments'][n]['pooled']['official'],8),
                  [round(entries[n][c]['score']['official']-state['incumbent']['cuts'][c]['score']['official'],6) for c in DEV],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('family',choices=('spectral','regime','regional','ensemble','audit')); run(p.parse_args())
