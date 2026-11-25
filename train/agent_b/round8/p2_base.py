"""Read-only replay of A's R4 release, in A's import context."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


def restore(out):
    from train.agent_a.round3 import submit_final as r3
    from train.agent_a.round6.competitive import residual_predict
    from train.agent_a.round8.protocol import contract, read, sha
    from train.shared.data import load_dataset
    from train.shared.preprocess import Normalizer, temporal_windows

    torch.set_num_threads(2)
    h,p=contract(); root=Path(p['paths']['root']); freeze=root/'integration/FROZEN.json'
    approval=read(freeze.with_name('P2-C3.json')); req=read(freeze.with_name('REQUEST_B_FULL_TRAIN.json'))
    assert req['protocol_sha256']==h and approval['passed'] and approval['frozen_sha256']==req['frozen_sha256']==sha(freeze)
    data=Path(p['paths']['data']); d=load_dataset(data)
    home=Path('.private/submissions/round5-v2'); old=read(home/'manifest.json')
    assert old['choice']=='R4' and old['weights']=={'r4':.25,'r5':0}
    assert sha(home/'r4-oof.pt')==old['artifacts_sha256']['r4-oof.pt']
    m=read(r3.OUT/'manifest.json'); recipe,*_=r3.frozen_recipe(); assert recipe==m['recipe']
    for n,v in m['source_sha256'].items(): assert sha(Path(n))==v
    for n,v in m['checkpoint_sha256'].items(): assert sha(r3.OUT/'checkpoints'/n)==v
    assert sha(r3.OUT/'normalizer.npz')==m['normalizer_sha256']
    with np.load(r3.OUT/'normalizer.npz',allow_pickle=False) as z: norm=Normalizer(**dict(z))
    with np.load(data/'test.npz',allow_pickle=False) as z: x,stamps=z['X'],z['times'].astype(str)
    t=pd.DatetimeIndex(pd.to_datetime(stamps,format='%Y%m%d%H',utc=True)); hours=t.hour.to_numpy(dtype=np.int64)
    w,missing=temporal_windows(x,t); raw,ef=r3.predict_models(recipe,Path(p['paths']['gpu_lock']),d,norm,w,hours,missing)
    design=r3.basis(SimpleNamespace(times=t,x=x,static=d.static),SimpleNamespace(times=t,h=hours))
    members=[]
    for cfg in recipe['models'].values():
        group=cfg.get('base',cfg.get('arm'))
        pred=ef if group is None else np.mean([raw[f'{group}-{s}'] for s in cfg['seeds']],axis=0,dtype=np.float64).astype('float32')
        if 'correction' in cfg: pred=r3.correct(pred,design,cfg['correction']['C3'])
        members.append(pred)
    base=r3.blend(members,np.array(list(recipe['weights'].values())))
    base[...,2:]+=.25*residual_predict(w,hours,d.static,torch.load(home/'r4-oof.pt',weights_only=True,map_location='cpu'))
    np.savez_compressed(out,prediction=base,times=stamps)


if __name__=='__main__': restore(sys.argv[1])
