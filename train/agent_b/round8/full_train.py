"""Deliver P2's frozen R7 input; never select parameters or produce a CSV."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from train.agent_b.round4.run import read, sha, write
from train.agent_b.round7.run import SPECS
from train.agent_b.round7.models import kernel, krr
from train.shared.artifacts import Experiment, aligned_predictions
from train.shared.data import load_dataset
from train.shared.metrics import score_mask
from train.shared.preprocess import temporal_windows
from .control import ORIG, ROOT, artifact, protocol
from .frozen_r7 import KEY


COMMAND='python -B -m train.agent_b.round8.full_train'


def gate(p,h):
    file=Path(p['integration']['freeze']); digest=sha(file); f=read(file)
    req=read(file.with_name('REQUEST_B_FULL_TRAIN.json')); result=read(file.with_name('P2-C3.json'))
    p2=next(e for e in f['proposals'] if e['id']=='P2'); m=f['models'][KEY]; meta=m['metadata']
    part=next(e for e in p2['parts'] if e['model']==KEY)
    assert h==f['protocol_sha256']==req['protocol_sha256'] and digest==req['frozen_sha256']==result['frozen_sha256']
    assert result['passed'] and p2==result['proposal'] and req['requested'][KEY]==m and meta==p['r7_pending']
    assert meta['prediction_kind']=='blend' and meta['base']=='R4' and meta['channels']==part['channels']==['V']
    assert meta['inner_weight']==part['weight']==.25 and part['effective_kernel_weight']==.0625
    assert Path(req['response']).resolve()==(ROOT/'FULL_TRAIN.json').resolve()
    historical=read(ROOT/'C3.json'); assert historical['frozen_sha256']==digest and historical['protocol_sha256']==h
    src=historical['models'][KEY]; artifact(src['prediction'],ROOT); ck=torch.load(artifact(src['checkpoint'],ROOT),weights_only=True,map_location='cpu')
    spec=tuple(ck['spec']); params=tuple(ck['params'])
    assert spec==SPECS['clock-lags'] and params==tuple(src['parameters']['parameters'])
    return digest,meta,spec,params,part


def infer(ck,x,t):
    h=t.hour.to_numpy(); kind,lags,anomaly=ck['spec']; assert anomaly
    w,m=temporal_windows((x-ck['gfs_climate'].numpy()[h])/ck['scale'].numpy(),t,lags)
    a=torch.from_numpy(w.reshape(len(x),-1)); a/=a.shape[1]**.5
    a=torch.cat([a,torch.from_numpy(m.astype('float32'))*.10],1)
    angle=torch.from_numpy((h*np.pi/12).astype('float32')); hour=torch.stack([angle.sin(),angle.cos()],1)
    q=kernel(a,ck['anchors'],hour,ck['hours'],kind,ck['params'][0]); dual=ck['dual']
    pred=torch.cat([q@dual[:,j:j+320] for j in range(0,dual.shape[1],320)],dim=1)
    return pred.numpy().reshape(len(t),70,35,4)+ck['climate'].numpy()[h]


def main(replay):
    torch.set_num_threads(2); p,h=protocol(); digest,meta,spec,params,part=gate(p,h)
    dest=ROOT/'release'/digest/'P2-R7'; dest.mkdir(parents=True,exist_ok=True)
    target=ROOT/'FULL_TRAIN.json'; previous=read(target) if target.exists() else None
    replay=replay or previous is not None
    if not replay:
        assert datetime.now(timezone.utc)<datetime.fromisoformat(p['deadlines']['deadline_utc'])
    data=Path(p['paths']['data']); hashes={k:sha(data/k) for k in ('train.npz','static.npz')}
    assert hashes==read(Path(p['paths']['agent_a'])/'state.json')['data_sha256']
    d=load_dataset(data); assert len(d.times)==2935
    cp=dest/'best.pt'; stats=dest/'normalizer.npz'; fit_record=dest/'fit.json'
    provenance=dict(training_inputs_sha256=hashes,frozen_sha256=digest,train_hours=len(d.times),spec=list(spec),parameters=list(params))
    def record(path): return dict(path=str(path),sha256=sha(path))
    if not fit_record.exists():
        if replay: raise ValueError('No committed fit to replay')
        tr=np.arange(len(d.times)); _,_,ck=next(krr(d,tr,tr[-1:],spec,(params,)))
        torch.save(ck,cp); np.savez(stats,**{k:ck[k].numpy() for k in ('gfs_climate','scale','climate')})
        write(fit_record,dict(**provenance,checkpoint=record(cp),statistics=record(stats)))
    fit=read(fit_record)
    for key in provenance: assert fit[key]==provenance[key] or key=='spec' and tuple(fit[key][1])==spec[1] and fit[key][::2]==list(spec)[::2]
    ck=torch.load(artifact(fit['checkpoint'],dest),weights_only=True,map_location='cpu'); artifact(fit['statistics'],dest)
    assert tuple(ck['spec'])==spec and tuple(ck['params'])==params and len(ck['anchors'])==2935
    # Only now open test; X/timestamps are exclusively used for inference.
    test_sha=sha(data/'test.npz')
    with np.load(data/'test.npz',allow_pickle=False) as z: x,stamps=z['X'],z['times'].astype(str)
    t=pd.DatetimeIndex(pd.to_datetime(stamps,format='%Y%m%d%H',utc=True))
    assert x.shape==(792,17,13,4) and x.dtype==np.float32 and np.isfinite(x).all()
    assert t.is_monotonic_increasing and not t.has_duplicates and t[0]-d.times[-1]>pd.Timedelta(hours=max(spec[1]))
    home=ORIG/'.private/submissions/round5-v2'; csv=home/'submission.csv'; base_csv_sha=sha(csv)
    assert base_csv_sha==read(home/'manifest.json')['csv_sha256']
    base_file=dest/'base-r4.npz'; base_receipt=dest/'base.json'
    if not base_receipt.exists():
        if replay: raise ValueError('No verified base replay')
        temp=dest/'base-r4.pending.npz'
        subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('p2_base.py').resolve()),str(temp)],cwd=ORIG,check=True,
                       env={**__import__('os').environ,'PYTHONPATH':str(ORIG)})
        temp.replace(base_file)
    with np.load(base_file,allow_pickle=False) as z: base=z['prediction']; np.testing.assert_array_equal(z['times'],stamps)
    assert base.shape==(792,70,35,4) and base.dtype==np.float32 and np.isfinite(base).all()
    archive=pd.read_csv(csv,dtype=str); id_parts=archive.id.str.split('_',expand=True)
    ti=pd.Index(stamps).get_indexer(id_parts[0]); rr=id_parts[1].to_numpy(dtype=int); cc=id_parts[2].to_numpy(dtype=int)
    mask=score_mask(t)
    assert (ti>=0).all() and ((rr>=0)&(rr<70)&(cc>=0)&(cc<35)).all()
    np.testing.assert_array_equal(np.sort(ti*2450+rr*35+cc),(np.flatnonzero(mask)[:,None]*2450+np.arange(2450)).ravel())
    cols=['RH','T','U','V']; exact=archive[cols].to_numpy(dtype=np.float32); scored=base[ti,rr,cc]; scored[:,0]=scored[:,0].clip(0,100)
    np.testing.assert_allclose(scored,exact,atol=5e-5,rtol=1e-6)
    base[ti,rr,cc]=exact
    verification=dict(base=record(base_file),base_csv_sha256=base_csv_sha,test_sha256=test_sha,max_scored_replay_error=float(np.abs(scored-exact).max()))
    if base_receipt.exists(): assert read(base_receipt)==verification
    else: write(base_receipt,verification)
    def blend(loaded):
        raw=infer(loaded,x,t); pred=base.copy(); pred[...,3]+=.25*(raw[...,3]-base[...,3]); return pred
    pred=blend(ck); np.testing.assert_array_equal(pred,blend(torch.load(cp,weights_only=True,map_location='cpu')))
    np.testing.assert_array_equal(pred[...,:3],base[...,:3]); assert np.isfinite(pred).all()
    out=object.__new__(Experiment); out.path=dest; file=dest/'prediction.npz'
    if not file.exists() and not replay:
        out.predictions('prediction',pred,t,d.static,mask,t[0].isoformat(),(t[-1]+pd.Timedelta(hours=1)).isoformat())
    saved,(arr,)=aligned_predictions([file]); np.testing.assert_array_equal(arr,pred)
    for k,v in [('times',stamps),('scored',mask),('wrf_lats',d.static['wrf_lats']),('wrf_lons',d.static['wrf_lons'])]: np.testing.assert_array_equal(saved[k],v)
    np.testing.assert_array_equal(arr[ti,rr,cc,:3],exact[:,:3])
    assert gate(p,h)[0]==digest
    manifest=dict(protocol_sha256=h,frozen_sha256=digest,training_inputs_sha256=hashes,test_sha256=test_sha,
                  models={KEY:dict(prediction=record(file),checkpoint=record(cp),statistics=record(stats),frozen_metadata=meta,
                  train_hours=2935,base_csv_sha256=base_csv_sha,base=verification,prediction_kind='blend',channels=['V'],
                  inner_weight=.25,p2_outer_weight=part['weight'],effective_kernel_weight=.0625,parameters=list(params),
                  reproduce=COMMAND+' --replay',inference_replayed=True)})
    if previous is not None: assert manifest==previous
    else: write(target,manifest)
    print(f'P2 R7 verified: {len(t)} hours, {mask.sum()} scored; {file}; sha256={sha(file)}',flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--replay',action='store_true'); main(a.parse_args().replay)
