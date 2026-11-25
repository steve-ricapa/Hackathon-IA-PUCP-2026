"""Answer A's explicitly frozen R7 historical request; not a development command."""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from train.agent_b.round4.run import Data, ROOT as R4, frozen_config, read, sha, write
from train.agent_b.round7.run import fit
from train.shared.artifacts import Experiment
from train.shared.metrics import score_mask
from .control import ROOT, artifact, protocol
from .run import compare_alignment


KEY='R7:channel-oof-wind'


def request(p,h):
    freeze=Path(p['integration']['freeze']); path=freeze.with_name('REQUEST_B_C3.json')
    if not freeze.is_file() or not path.is_file():
        raise ValueError('No coordinated C3 request from A')
    digest=sha(freeze); proposal=read(freeze); req=read(path)
    if req['protocol_sha256']!=h or proposal['protocol_sha256']!=h or req['frozen_sha256']!=digest:
        raise ValueError('C3 request does not match freeze and protocol')
    if KEY not in req['requested'] or not any(part['model']==KEY for row in proposal['proposals'] for part in row['parts']):
        raise ValueError('Frozen R7 candidate was not requested')
    if proposal['models'][KEY]!=req['requested'][KEY] or proposal['models'][KEY]['metadata']!=p['r7_pending']:
        raise ValueError('R7 recipe differs from frozen handoff')
    return digest


def main():
    p,h=protocol(); digest=request(p,h)
    if datetime.now(timezone.utc)>=datetime.fromisoformat(p['deadlines']['deadline_utc']):
        raise ValueError('Shared campaign deadline reached')
    torch.set_num_threads(2); frozen_config(); base_link=read(R4/'final.json')['predictions']
    base_path=artifact(base_link,R4); cut='C3'; d=Data(); _,va,_,t,bounds,*_=d.fold(cut)
    assert dict(start=bounds[0],end_exclusive=bounds[1])==p['folds'][cut]
    base=compare_alignment(d.ds,va,bounds,base_path)
    raw,ck,info=fit(d,cut,'clock-lags',None)
    pred=base.copy(); pred[...,3]+=.25*(raw[...,3]-base[...,3])
    dest=ROOT/'frozen'/digest/'R7'; out=Experiment(dest)
    out.predictions('prediction',pred,t,d.ds.static,score_mask(t),*bounds)
    cp=dest/'best.pt'; torch.save(ck,cp); file=dest/'prediction.npz'
    np.testing.assert_array_equal(compare_alignment(d.ds,va,bounds,file),pred)
    assert request(p,h)==digest
    response=ROOT/'C3.json'; doc=read(response) if response.exists() else dict(protocol_sha256=h,frozen_sha256=digest,models={})
    assert doc['protocol_sha256']==h and doc['frozen_sha256']==digest
    doc['models'][KEY]=dict(prediction=dict(path=str(file),sha256=sha(file)),base_sha256=sha(base_path),
        checkpoint=dict(path=str(cp),sha256=sha(cp)),train_end_exclusive=bounds[0],parameters=info,
        prediction_kind='blend',channels=['V'],inner_weight=.25,
        command='python -B -m train.agent_b.round8.frozen_r7')
    write(response,doc)


if __name__=='__main__': main()
