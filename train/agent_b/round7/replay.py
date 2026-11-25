"""Verify a full-field KRR checkpoint by predicting without fitting."""
import argparse

import numpy as np
import torch

from train.agent_b.round4.run import read, sha
from train.agent_b.round5.loop import Development
from train.shared.preprocess import temporal_windows
from .models import kernel
from .run import DEV, ROOT, load, protocol


def replay(name):
    torch.set_num_threads(2); _,digest,_=protocol(); state=read(ROOT/'research_state.json')
    assert state['protocol_sha256']==digest
    d=Development(); ds=d.ds
    for cut in DEV:
        e=state['experiments'][name]['cuts'][cut]; link=e['checkpoint']; p=ROOT/link['path']; assert sha(p)==link['sha256']
        ck=torch.load(p,map_location='cpu',weights_only=True)
        kind,lags,anomaly=ck['spec']; _,va,*_=d.fold(cut); stop=va[-1]+1
        t=ds.times[:stop]; h=t.hour.to_numpy(); x=ds.x[:stop]
        if anomaly: x=x-ck['gfs_climate'].numpy()[h]
        w,m=temporal_windows(x/ck['scale'].numpy(),t,lags)
        a=torch.from_numpy(w.reshape(stop,-1)); a/=a.shape[1]**.5
        a=torch.cat([a,torch.from_numpy(m.astype('float32'))*.10],1)
        angle=torch.from_numpy((h*np.pi/12).astype('float32')); hour=torch.stack([angle.sin(),angle.cos()],1)
        query=kernel(a[va],ck['anchors'],hour[va],ck['hours'],kind,ck['params'][0]); dual=ck['dual']
        chunks=[(query@dual[:,i:i+320]).numpy() for i in range(0,dual.shape[1],320)]
        pred=np.concatenate(chunks,1).reshape(len(va),*ds.y.shape[1:])+ck['climate'].numpy()[h[va]]
        np.testing.assert_allclose(pred,load(d,cut,e),atol=3e-5,rtol=1e-6)
        print(name,cut,'checkpoint replay OK')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('model',choices=('linear','rbf','clock','clock-lags')); replay(p.parse_args().model)
