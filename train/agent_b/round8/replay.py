"""Replay an audited development checkpoint into a caller-selected directory."""
import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from train.agent_b.round4.run import read, sha
from train.shared.artifacts import Experiment
from train.shared.data import load_dataset, split_indices
from train.shared.metrics import score_mask
from train.shared.preprocess import Normalizer
from .control import ORIG, ROOT, artifact, protocol, source
from .run import ARMS, compare_alignment, inputs, model_for, predict


def replay(arm,cut,seed,out,device):
    p,h=protocol(); directory=ROOT/'runs'/arm/f's{seed}'/cut
    record=read(directory/'summary.json')
    if (record['protocol_sha256'],record['arm'],record['cut'],record['seed'])!=(h,arm,cut,seed):
        raise ValueError('Run provenance mismatch')
    paths={k:artifact(record[k],directory) for k in ('checkpoint','normalizer','prediction')}
    ds=load_dataset(p['paths']['data']); fold=p['folds'][cut]
    b=(fold['start'],fold['end_exclusive']); _,va=split_indices(ds.times,*b)
    with np.load(paths['normalizer'],allow_pickle=False) as z: norm=Normalizer(**dict(z))
    ck=torch.load(paths['checkpoint'],map_location='cpu',weights_only=True)
    if ck['train_end_exclusive']!=b[0] or ck['config']!=ARMS[arm] or ck['seed']!=seed:
        raise ValueError('Checkpoint provenance mismatch')
    saved=compare_alignment(ds,va,b,paths['prediction']); x=inputs(ds,norm,va[-1]+1)
    model=model_for(ds,ARMS[arm],torch.load(artifact(record['control_source']['checkpoint'],directory),map_location='cpu',weights_only=True))
    model.load_state_dict(ck['state_dict']); gpu=device=='cuda'
    lock=source('r8_replay_gpu',ORIG/'train/agent_a/gpu.py').gpu_lock
    with (lock(Path(p['paths']['gpu_lock'])) if gpu else nullcontext()):
        model.to(device); x=tuple(v.to(device) for v in x)
        try: pred=predict(model,x,va,norm)
        finally:
            del x,model
            if gpu: torch.cuda.synchronize(); torch.cuda.empty_cache()
    np.testing.assert_allclose(pred,saved,rtol=1e-5,atol=2e-4)
    export=Experiment(out); t=ds.times[va]
    export.predictions('prediction',pred,t,ds.static,score_mask(t),*b)
    export.json('replay',dict(protocol_sha256=h,source=record['checkpoint'],device=device,
        sha256=sha(out/'prediction.npz'),max_abs_error=float(np.abs(pred-saved).max())))


if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('arm',choices=ARMS); a.add_argument('cut',choices=('C0','C1','C2'))
    a.add_argument('--seed',type=int,choices=(0,1,2),default=0); a.add_argument('--output',type=Path,required=True)
    a.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    r=a.parse_args(); torch.set_num_threads(2); replay(r.arm,r.cut,r.seed,r.output,r.device)
