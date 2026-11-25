"""Summarize paired development scores without opening any validation labels."""
from collections import defaultdict

from train.agent_b.round4.run import DEV, ROOT as R4, read, sha
from train.shared.artifacts import aligned_predictions

from .loop import ROOT, Development, INCUMBENT, nested, ranking


def summarize(state):
    incumbent=state['incumbent']
    groups=defaultdict(list)
    for name,result in state['experiments'].items():
        groups[result['family']].append((name,result))
    rows=[]
    for family,entries in groups.items():
        name,best=max(entries,key=lambda x:x[1]['pooled']['official'])
        deltas=[best['cuts'][c]['score']['official']-incumbent['cuts'][c]['score']['official'] for c in DEV]
        rows.append(dict(family=family,count=len(entries),name=name,pooled=best['pooled'],deltas=deltas))
    name,value,folds,weeks=ranking(state)
    entry=state['experiments'].get(name,incumbent)
    blocks=[]
    for c in DEV:
        new=entry['cuts'][c]['slices']; old=incumbent['cuts'][c]['slices']
        delta=[v['official']-old[k]['official'] for k,v in new.items() if k.startswith('(2,')]
        blocks.append(dict(cut=c,positive=sum(d>1e-8 for d in delta),total=len(delta),deltas=delta))
    return dict(families=rows,winner=dict(name=name,official=value,pooled=entry['pooled'],
             fold_deltas=folds,week_deltas=weeks,paired_blocks=blocks),nested=nested(state))


def replay_fixed(state):
    """Recreate the promoted fixed wind mixture without fitting any model."""
    import numpy as np
    import torch
    from .spectral import decode, design

    torch.set_num_threads(3)
    name=ranking(state)[0]
    run=state['experiments'][name]; source=state['experiments'][run['spec']['fixed']]
    d=Development(); d.previous=read(R4/'research_state.json')['experiments'][INCUMBENT]['cuts']
    for cut in DEV:
        tr,va,_,_,_,_,n,c=d.fold(cut)
        link=source['cuts'][cut]['checkpoint']; path=ROOT/link['path']
        assert sha(path)==link['sha256']
        fit=torch.load(path,map_location='cpu',weights_only=True)
        _,v,mean,sd=design(d,tr,va,n,source['spec'])
        torch.testing.assert_close(mean,fit['mean'],atol=0,rtol=0)
        torch.testing.assert_close(sd,fit['sd'],atol=0,rtol=0)
        p=decode(v,fit['coef'],fit['intercept'],source['spec'],n,c)
        base=d.incumbent(cut); direct=base+source['weight']*(p-base)
        intermediate=base.copy(); intermediate[...,2:]=direct[...,2:]
        reconstructed=base+run['weight']*(intermediate-base)
        _,(saved,)=aligned_predictions([ROOT/run['cuts'][cut]['prediction']['path']])
        np.testing.assert_array_equal(saved,reconstructed)
        np.testing.assert_array_equal(saved[...,:2],base[...,:2])
    print(f'Replayed {name} bit for bit on {len(DEV)} cuts')


if __name__=='__main__':
    import argparse
    import json
    parser=argparse.ArgumentParser(); parser.add_argument('--replay-fixed',action='store_true')
    state=read(ROOT/'research_state.json')
    if parser.parse_args().replay_fixed: replay_fixed(state)
    else: print(json.dumps(summarize(state),indent=2))
