import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
import torch

from train.shared.artifacts import Experiment, aligned_predictions
from train.shared.data import load_dataset, split_indices
from train.shared.metrics import OFFICIAL_SCALE, evaluate, score_mask
from train.shared.preprocess import Normalizer, temporal_windows
from .models import eof_wind, features, local_design, local_regression

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]/'.private/experiments/round4/agent_b'
ORIG=HERE.parents[3]/'Hackathon-IA-PUCP-2026'
R3=ORIG/'.private/experiments/round3'
DEV=('C0','C1','C2')
SPECS={
 'kernel':dict(kind='kernel',modes=16,lags=(2,1,0),ridge=5.),
 'long-lags':dict(kind='rrr',modes=16,lags=(24,12,6,2,1,0)),
 'terrain':dict(kind='terrain',lags=(2,1,0),ridge=.25),
 'dynamics':dict(kind='dynamics',lags=(2,1,0),ridge=.3),
 'oof':dict(kind='oof',lags=(2,1,0),ridge=2.),
 'oof-smooth':dict(kind='oof',lags=(2,1,0),ridge=2.,smooth=5),
 'oof-hour':dict(kind='oof',lags=(2,1,0),ridge=2.,hour_only=True),
 'oof-v':dict(kind='oof',lags=(2,1,0),ridge=2.,v_only=True),
 'oof-shrunk':dict(kind='oof',lags=(2,1,0),ridge=8.),
 'kernel-tight':dict(kind='kernel',modes=8,lags=(2,1,0),ridge=5.),
}
WEIGHTS=(.25,.5)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8'); temp.replace(path)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def frozen_config():
    freeze=read(ROOT/'best_config.json')
    for key,path in [('protocol_sha256',R3/'protocol.json'),
                     ('reference_sha256',R3/'integration/frozen.json'),
                     ('development_sha256',ROOT/'research_state.json')]:
        if sha(path)!=freeze[key]:
            raise ValueError(f'Frozen input hash mismatch: {key}')
    return freeze


def pool(m):
    e=np.sqrt(np.average(np.square([x['rmse'] for x in m]),weights=[x['hours'] for x in m],axis=0))
    return dict(official=float(1-(e/OFFICIAL_SCALE).mean()),rmse=e.tolist(),hours=sum(x['hours'] for x in m))


class Data:
    def __init__(self):
        self.ds=load_dataset(ORIG/'data')
        self.bounds=read(R3/'protocol.json')['folds']
        self.weights=read(R3/'integration/frozen.json')['weights']
        self.cache={}

    def fold(self,cut):
        if cut in self.cache: return self.cache[cut]
        d=self.ds; b=self.bounds[cut]; bounds=(b['start'],b['end_exclusive']); tr,va=split_indices(d.times,*bounds)
        stage='DONE' if cut=='C3' else 'DEVELOPMENT'
        manifests={k:read(R3/f'agent_{k.lower()}'/f'{stage}.json')['models'] for k in 'AB'}
        p,ww={},[]
        for key, w in self.weights.items():
            tag,name=key.split('/',1)
            rel=manifests[tag][name]['predictions'].get(cut)
            if rel is not None:
                p[R3/f'agent_{tag.lower()}'/rel]=None; ww.append(w)
            elif cut!='C0': raise ValueError('Reference incomplete')
        meta,arrays=aligned_predictions(list(p)); t=d.times[va]
        for key,value in [('times',t.strftime('%Y%m%d%H')),('scored',score_mask(t)),('wrf_lats',d.static['wrf_lats']),('wrf_lons',d.static['wrf_lons'])]:
            np.testing.assert_array_equal(meta[key],value)
        ww=np.asarray(ww); ww=ww/ww.sum(0) if cut=='C0' else ww
        base=np.zeros_like(arrays[0],dtype=np.float64)
        for arr,w in zip(arrays,ww):base+=arr*w
        h=d.times.hour.to_numpy(dtype=np.int64)
        n=Normalizer.fit(d.x[tr],d.y[tr],h[tr]); c=n.clim[h[va]]
        tup=(tr,va,h,t,bounds,base.astype(np.float32),n,c)
        self.cache[cut]=tup
        return tup


def candidate(data,cut,spec):
    ds=data.ds; tr,va,h,t,bounds,base,n,c=data.fold(cut)
    stop=va[-1]+1
    w,m=temporal_windows(ds.x[:stop],ds.times[:stop],tuple(spec['lags']))
    kind=spec['kind']
    if kind=='oof':
        # Use only complete historical OOF folds; C0 has no previous reference.
        earlier=[k for k in DEV if data.bounds[k]['end_exclusive']<=bounds[0]]
        if not earlier:return base[...,2:].copy(),dict(identity=True)
        indices=[];truth=[]
        for k in earlier:
            _,v,_,_,_,p,_,_=data.fold(k)
            indices.extend(v); truth.append(ds.y[v,...,2:]-p[...,2:])
        a=local_design(w[indices],h[indices],ds.static,'dynamics');v=local_design(w[va],h[va],ds.static,'dynamics')
        if spec.get('hour_only'):
            a,v=a[...,8:10],v[...,8:10]
        pred,fit=local_regression(a,v,np.concatenate(truth),spec['ridge']);
        delta=pred.reshape(*base.shape[:-1],2)
        if spec.get('smooth'):
            from torch.nn import functional as F
            d=torch.from_numpy(delta).permute(0,3,1,2)
            k=spec['smooth']; p=k//2
            delta=F.avg_pool2d(F.pad(d,(p,p,p,p),mode='replicate'),k,stride=1).permute(0,2,3,1).numpy()
        if spec.get('v_only'):delta[...,0]=0
        return base[...,2:]+delta,fit
    y=n.targets(ds.y[tr],h[tr])
    if kind in ('kernel','rrr'):
        a=features(n.inputs(w[tr]),h[tr],m[tr],ds.static); v=features(n.inputs(w[va]),h[va],m[va],ds.static)
        pred,fit=eof_wind(a,v,y,h[tr],h[va],spec)
    else:
        a=local_design(w[tr],h[tr],ds.static,kind);v=local_design(w[va],h[va],ds.static,kind)
        pred,fit=local_regression(a,v,y[...,2:],spec['ridge']);pred=pred.reshape(*base.shape[:-1],2)
    return c[...,2:]+pred*n.anom_sd[2:],fit


def measure(data,cut,pred):
    _,va,_,t,_,_,_,c=data.fold(cut)
    return evaluate(pred,data.ds.y[va],t,c)


def blocks(data,cut,pred):
    _,va,_,t,_,base,_,c=data.fold(cut)
    days=t.normalize().unique(); out=[]
    for i in range(0,len(days),2):
        keep=t.normalize().isin(days[i:i+2]); truth=data.ds.y[va[keep]]
        a=evaluate(pred[keep],truth,t[keep],c[keep]);b=evaluate(base[keep],truth,t[keep],c[keep]);
        out.append(dict(start=days[i].date().isoformat(),hours=a['hours'],delta=a['official']-b['official']))
    return out


def save(data,cut,name,pred,fit,metadata):
    _,_,_,t,b,base,n,_=data.fold(cut); out=Experiment(ROOT/'runs'/name/cut)
    np.testing.assert_array_equal(pred[...,:2],base[...,:2])
    out.predictions('predictions',pred,t,data.ds.static,score_mask(t),*b)
    torch.save(fit,out.path/'best.pt'); np.savez(out.path/'normalizer.npz',**vars(n))
    _,(p,)=aligned_predictions([out.path/'predictions.npz']);np.testing.assert_array_equal(p,pred)
    metrics=measure(data,cut,p); assert metrics==measure(data,cut,pred)
    obj=dict(metrics=metrics,blocks=blocks(data,cut,p),**metadata)
    for key,f in [('predictions','predictions.npz'),('checkpoint','best.pt'),('normalizer','normalizer.npz')]:
        q=out.path/f;obj[key]=dict(path=q.relative_to(ROOT).as_posix(),sha256=sha(q))
    out.json('summary',obj)
    return obj


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('develop','final'));parser.add_argument('--only',choices=list(SPECS),nargs='+');args=parser.parse_args()
    torch.set_num_threads(3); data=Data(); started=perf_counter()
    if args.stage=='develop':
        if (ROOT/'best_config.json').exists():
            raise FileExistsError('Round 4 is frozen; preserve the development snapshot.')
        f=ROOT/'research_state.json'
        state=read(f) if f.exists() else dict(baselines={},experiments={})
        state['baselines']={k:measure(data,k,data.fold(k)[5]) for k in DEV}
        baseline=pool(list(state['baselines'].values()))
        print('reference',baseline,flush=True)
        for kind in args.only or SPECS:
            spec=SPECS[kind]
            for cut in DEV:
                if all((ROOT/'runs'/f'{kind}-{w}'/cut/'summary.json').exists() for w in WEIGHTS):continue
                t=perf_counter();wind,fit=candidate(data,cut,spec);elapsed=perf_counter()-t
                mm=psutil.Process().memory_info();ram=getattr(mm,'peak_wset',mm.rss)/2**20
                for w in WEIGHTS:
                    name=f'{kind}-{w}'; file=ROOT/'runs'/name/cut/'summary.json'
                    if file.exists():continue
                    base=data.fold(cut)[5]; pred=base.copy();pred[...,2:]=base[...,2:]+w*(wind-base[...,2:]);
                    save(data,cut,name,pred,fit,dict(seconds=elapsed,peak_ram_mib=ram,config=spec,weight=w))
            for w in WEIGHTS:
                name=f'{kind}-{w}'; entries={c:read(ROOT/'runs'/name/c/'summary.json') for c in DEV}; summary=pool([z['metrics'] for z in entries.values()]);
                state['experiments'][name]=dict(config=spec,weight=w,pooled=summary,cuts=entries)
                print(name,summary['official'],[round(x['metrics']['official']-state['baselines'][c]['official'],6) for c,x in entries.items()],flush=True)
            state['runtime_last_stage_seconds']=perf_counter()-started;write(f,state)
    else:
        freeze=frozen_config(); name=freeze['name'];cut='C3'; data.fold(cut)
        base=data.fold(cut)[5]
        np.testing.assert_allclose(measure(data,cut,base)['official'],read(R3/'integration/report.json')['chosen_C3']['official'],atol=1e-8,rtol=0)
        pred=base.copy();fit=dict(identity=True)
        if name!='reference':
            wind,fit=candidate(data,cut,freeze['config']); pred[...,2:]+=freeze['weight']*(wind-base[...,2:])
        obj=save(data,cut,name,pred,fit,dict(seconds=perf_counter()-started,reference=measure(data,cut,base)))
        write(ROOT/'final.json',obj); print(obj['metrics'],flush=True)


if __name__=='__main__':main()
