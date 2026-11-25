"""Audit seeds, fixed wind blends, and prepare A's development intake."""
import argparse
import json

import numpy as np

from train.agent_b.round4.run import pool, read, sha, write
from train.agent_b.round5.loop import Development, ROOT as R5
from train.shared.artifacts import aligned_predictions
from train.shared.metrics import evaluate
from .run import R6, ROOT, assert_reference, entry, protocol, reference, verify


def report(publish=False):
    proto,digest=protocol(); dev=proto['development']; state=read(ROOT/'research_state.json')
    assert state['protocol_sha256']==digest and state['reference_state_sha256']==sha(R5/'research_state.json')
    d=Development(); r5=read(R5/'research_state.json'); base={c:reference(d,c,r5) for c in dev}
    baseline={}; rows=[]
    for name,e in state['experiments'].items():
        predictions={}
        for c,s in e['cuts'].items():
            verify(d,c,s,base[c]); meta,(p,)=aligned_predictions([ROOT/s['prediction']['path']]); assert_reference(meta,d,c)
            predictions[c]=p; baseline[c]=s['evaluation']['reference']
        for weight in (.25,.5,1.):
            cuts={}
            for c in dev:
                _,va,_,t,_,_,_,clim=d.fold(c)
                p=base[c]+weight*(predictions[c]-base[c]) if weight!=1 else predictions[c]
                cuts[c]=evaluate(p,d.ds.y[va],t,clim)
            rows.append(dict(name=name,weight=weight,cuts=cuts,pooled=pool(list(cuts.values())),
                             pooled_c1c2=pool([cuts[c] for c in dev[1:]]),
                             delta={c:cuts[c]['official']-baseline[c]['official'] for c in dev}))
    if not rows: raise ValueError('No fully verified configs to publish')
    rows.sort(key=lambda e:e['pooled']['official'],reverse=True)
    comparator=pool(list(baseline.values())); nested={}
    for i,c in enumerate(dev[1:],1):
        old=dev[:i]; best=max(rows,key=lambda r:pool([r['cuts'][x] for x in old])['official'])
        use=pool([best['cuts'][x] for x in old])['official']>pool([baseline[x] for x in old])['official']
        nested[c]=dict(selected=best['name'] if use else 'reference',weight=best['weight'] if use else 0.,
                       delta=best['delta'][c] if use else 0.)
    chosen=next((r for r in rows if r['pooled']['official']>comparator['official'] and min(r['delta'].values())>=0),None)
    if chosen:
        model=state['experiments'][chosen['name']]
        peers=[r for r in rows if r['weight']==chosen['weight'] and
               state['experiments'][r['name']]['architecture']==model['architecture'] and
               state['experiments'][r['name']]['max_epochs']==model['max_epochs']]
        chosen['confirmation']={str(state['experiments'][r['name']]['seed']):r['pooled']['official']-comparator['official']
                                for r in peers}
        chosen['confirmed']=all(chosen['confirmation'].get(str(s),-1.)>0 for s in proto['seeds']['confirmation'])
    chosen=chosen or dict(name='reference',weight=0.,pooled=comparator)
    write(ROOT/'best_config.json',dict(selection=chosen,screened=rows[0],reference=comparator,nested=nested,
                                     protocol_sha256=digest,c3='not accessed'))
    write(ROOT/'blend_report.json',dict(rows=rows,baseline=baseline,nested=nested))
    if publish:
        models={}
        # Supply only the best eligible specialist; never use negative screens as finalists.
        if chosen['name']!='reference' and chosen['confirmed']:
            k=chosen['name']; e=state['experiments'][k]
            models[k]=dict(predictions={c:s['prediction']['path'] for c,s in e['cuts'].items()},
                 metadata=dict(wind_only=True,channels=['U','V'],architecture=e['architecture'],seed=e['seed'],
                    seed_count=len(chosen['confirmation']),blend_suggestion=chosen['weight'],historical_generalization_unverified=True,
                    metrics=chosen,artifacts={c:{x:s[x] for x in ('prediction','checkpoint')} for c,s in e['cuts'].items()}))
        manifest=dict(protocol_sha256=digest,models=models,status='development_only',
                      c3='not scored; requires coordinated freeze',reference=comparator,screened_best=rows[0],
                      configurations=len(state['experiments']),nested_replay=nested)
        write(ROOT/'DEVELOPMENT.json',manifest)
        write(ROOT/'DONE.json',manifest) # READY explicitly permits dev-only intake for B.
    print(json.dumps(dict(best=chosen,top=rows[:4],nested=nested),indent=2))


if __name__=='__main__':
    args=argparse.ArgumentParser(); args.add_argument('--publish',action='store_true'); report(args.parse_args().publish)
