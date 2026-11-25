"""Incremental paired screening, confirmation and atomic handoff."""
import argparse
import os
import time
from datetime import datetime, timezone

from train.agent_b.round4.run import pool, read, sha, write
from train.shared.artifacts import aligned_predictions
from train.shared.data import load_dataset, split_indices
from train.shared.preprocess import Normalizer
from .run import ARMS, control, report_fold, run
from .control import ROOT, Pending, protocol

import numpy as np
import torch

DEV=('C0','C1','C2')
FIRST=('B1-60','B2-0-100','B2-30-100','B2-0-300','B2-30-300','B3-shared','B3-local-low','B3-local-high')


def control_arm(arm):
    if arm.startswith(('B2-30','B2-60')): return f'B2-0-{ARMS[arm]["steps"]}'
    if arm.startswith('B3-local'): return 'B3-shared'
    return None


def paired(a,b=None):
    ca=[x['candidate'] for x in a]; co=[x['candidate' if b else 'control'] for x in (b or a)]
    weeks=[(pair[0],(b[i]['weeks'][j][0] if b else pair[1])) for i,e in enumerate(a) for j,pair in enumerate(e['weeks'])]
    scores=[pool(ca),pool(co)]; deltas=[x['official']-y['official'] for x,y in zip(ca,co)]
    leave=[pool([x for j,(x,y) in enumerate(weeks) if j!=i])['official']-
           pool([y for j,(x,y) in enumerate(weeks) if j!=i])['official'] for i in range(len(weeks))]
    gain=scores[0]['official']-scores[1]['official']
    return dict(candidate=scores[0],control=scores[1],gain=gain,deltas=deltas,leave_one_week=leave,
                promoted=gain>0 and sum(x>0 for x in deltas)>=2 and min(leave)>0)


def digest_r7(p):
    from pathlib import Path
    r=p['r7_pending']; src=Path(r['source_manifest']); assert sha(src)==r['source_manifest_sha256']
    for a in r['artifacts'].values(): assert sha(src.parent/a['path'])==a['sha256']
    write(ROOT/'R7_HANDOFF.json',dict(**r,semantics='C0 exact R4; C1/C2 V=R4_V+0.25*(clock-lags_V-R4_V); RH,T,U exact R4; full stored blend, do not apply 0.25 again.'))


def summary(rows,arm,seed):
    return [rows[arm][str(seed)][c]['evaluation'] for c in DEV]


def publish(p,digest,state,done=False):
    rows=state['results']; screening={}; confirmed={}; models={}
    complete=lambda a,s: set(rows.get(a,{}).get(str(s),{}))==set(DEV)
    for arm in rows:
        peer=control_arm(arm)
        if not complete(arm,0) or peer and not complete(peer,0): continue
        screening[arm]=paired(summary(rows,arm,0),summary(rows,peer,0) if peer else None)
    order=sorted((a for a,v in screening.items() if v['promoted'] and a not in ('B3-shared',) and not a.startswith('B2-0')),key=lambda a:screening[a]['gain'],reverse=True)
    for arm in order:
        peer=control_arm(arm); seeds=[s for s in range(3) if complete(arm,s) and (not peer or complete(peer,s))]
        if seeds==[0,1,2]:
            ds=load_dataset(p['paths']['data']); ev=[]
            for c in DEV:
                b=p['folds'][c]; tr,va=split_indices(ds.times,b['start'],b['end_exclusive']); n=Normalizer.fit(ds.x[tr],ds.y[tr],ds.times[tr].hour.to_numpy())
                def avg(a): return np.mean([aligned_predictions([x])[1][0] for x in a],axis=0,dtype=np.float64).astype('float32')
                def loc(a,s): return ROOT/'runs'/a/f's{s}'/c/rows[a][str(s)][c]['prediction']['path']
                pred=avg([loc(arm,s) for s in seeds]); base=avg([loc(peer,s) if peer else rows[arm][str(s)][c]['control_source']['prediction']['path'] for s in seeds])
                ev.append(report_fold(ds,va,n,pred,base,b['start']))
            confirmed[arm]=dict(**paired(ev),seed_scores=[paired(summary(rows,arm,s),summary(rows,peer,s) if peer else None) for s in seeds])
        for s in seeds:
            entries=rows[arm][str(s)]
            models[f'{arm}-s{s}']=dict(predictions={c:f'runs/{arm}/s{s}/{c}/{v["prediction"]["path"]}' for c,v in entries.items()},
                metadata=dict(configuration=ARMS[arm],seed=s,prediction_kind='full',prediction_type='full',artifacts=entries,
                              comparison=screening[arm],confirmation=confirmed.get(arm),
                              suggested_channels=[k for k,x,y in zip(('RH','T','U','V'),screening[arm]['candidate']['rmse'],screening[arm]['control']['rmse']) if x<y]))
    state.update(screening=screening,confirmed=confirmed,screening_order=order,
                 status='DONE' if done else 'CANDIDATE_READY' if models else 'RUNNING')
    write(ROOT/'research_state.json',state)
    write(ROOT/'best_config.json',dict(screening=order[0] if order else None,
          confirmed=max((a for a in confirmed if confirmed[a]['promoted']),key=lambda a:confirmed[a]['gain'],default=None),
          comparisons=screening,confirmation=confirmed))
    if models or done:
        write(ROOT/('DONE.json' if done else 'CANDIDATE_READY.json'),dict(protocol_sha256=digest,status=state['status'],models=models,
               screening=screening,confirmed=confirmed,c3='not accessed',research_state_sha256=sha(ROOT/'research_state.json')))
    return order


def main(watch):
    torch.set_num_threads(2); p,digest=protocol(); digest_r7(p); path=ROOT/'research_state.json'
    state=read(path) if path.exists() else dict(protocol_sha256=digest,results={},failures={})
    assert state['protocol_sha256']==digest
    if state.get('status')=='DONE': return
    now=lambda:datetime.now(timezone.utc)
    deadlines={k:datetime.fromisoformat(v) for k,v in p['deadlines'].items()}
    def persist(): write(path,state)
    def attempt(arm,seed,c):
        key=f'{arm}/s{seed}/{c}'
        if c in state['results'].get(arm,{}).get(str(seed),{}) or key in state['failures']: return False
        try: control(p,digest,c,seed)
        except Pending: return False
        state.update(process_id=os.getpid(),active=key,next_action='fit or verify'); persist()
        try:
            result=run(arm,c,seed)
            state['results'].setdefault(arm,{}).setdefault(str(seed),{})[c]=result
            print(key,result['evaluation']['candidate']['official'],flush=True)
        except Pending: return False
        except Exception as exc:
            state['failures'][key]=repr(exc); print(key,repr(exc),flush=True)
        finally:
            state['active']=None; persist()
        return True
    while True:
        if now()>=deadlines['confirmation_end_utc']:
            state.update(active=None,next_action='A integration only'); publish(p,digest,state,True); return
        ranked=publish(p,digest,state); queue=list(FIRST)
        b1=state['screening'].get('B1-60')
        if b1 and (b1['gain']>0 or max(b1['deltas'])>0): queue.extend(('B1-90','B1-30'))
        for n in (100,300):
            if f'B2-30-{n}' in ranked: queue.append(f'B2-60-{n}')
        jobs=[]
        # Confirm a positive signal as soon as its matching controls arrive, without ending screening.
        for arm in ranked[:2]:
            for seed in (1,2):
                jobs.extend((a,seed,c) for a in (control_arm(arm),arm) if a for c in DEV)
        if now()<deadlines['exploration_end_utc']:
            jobs.extend((arm,0,c) for arm in queue for c in DEV)
        state['queue']=jobs; persist()
        if any(attempt(*job) for job in jobs): continue
        state.update(active=None,next_action='Waiting for eligible A controls or handoff deadline; no GPU lease'); persist()
        if not watch: return
        time.sleep(60)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--watch',action='store_true'); main(p.parse_args().watch)
