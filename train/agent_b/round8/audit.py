"""Read-only rescore of persisted B runs against their pinned A controls."""
import argparse
from pathlib import Path

import numpy as np
import torch

from train.agent_b.round4.run import ORIG, read
from train.shared.data import load_dataset, split_indices
from train.shared.preprocess import Normalizer
from .control import ROOT, artifact, protocol
from .run import compare_alignment, report_fold


def audit(name=None):
    p,h=protocol(); state=read(ROOT/'research_state.json'); ds=load_dataset(p['paths']['data']); count=0
    if state['protocol_sha256']!=h: raise ValueError('Wrong state protocol')
    for arm,by_seed in state['results'].items():
        if name and arm!=name: continue
        for seed,by_cut in by_seed.items():
            for cut,e in by_cut.items():
                d=ROOT/'runs'/arm/f's{seed}'/cut; b=p['folds'][cut]; bounds=(b['start'],b['end_exclusive'])
                tr,va=split_indices(ds.times,*bounds)
                if (e['protocol_sha256'],e['arm'],e['seed'],e['cut'],e['train_end_exclusive'])!=(h,arm,int(seed),cut,bounds[0]):
                    raise ValueError('Inconsistent summary provenance')
                cp=artifact(e['checkpoint'],d); npath=artifact(e['normalizer'],d)
                ck=torch.load(cp,map_location='cpu',weights_only=True)
                assert (ck['seed'],ck['config'],ck['train_end_exclusive'])==(int(seed),e['configuration'],bounds[0])
                with np.load(npath,allow_pickle=False) as z: n=Normalizer(**dict(z))
                source=e['control_source']; refn=artifact(source['normalizer'],ORIG)
                with np.load(refn,allow_pickle=False) as z:
                    for key,v in vars(n).items(): np.testing.assert_array_equal(z[key],v)
                q=compare_alignment(ds,va,bounds,artifact(e['prediction'],d))
                r=compare_alignment(ds,va,bounds,artifact(source['prediction'],ORIG)); artifact(source['checkpoint'],ORIG)
                assert ds.times[tr[-1]].isoformat()==e['train_last']
                assert report_fold(ds,va,n,q,r,bounds[0])==e['evaluation']
                count+=1
    if not count: raise ValueError('No completed runs to audit')
    print(f'{count} R8 cut/seed runs verified')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--arm'); audit(parser.parse_args().arm)
