"""Bounded R8 continuation: confirm one localized readout, audit, hand off."""
import os
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch

from train.agent_b.round4.run import read, write
from .audit import audit
from .campaign import DEV, publish
from .control import ROOT, Pending, protocol
from .run import run


def main():
    torch.set_num_threads(2); p,h=protocol(); path=ROOT/'research_state.json'; s=read(path)
    assert s['protocol_sha256']==h
    if (ROOT/'DONE.json').exists():
        print('DONE already exists; no runs repeated.'); return
    pid=s.get('process_id')
    if pid and psutil.pid_exists(pid):
        cmd=psutil.Process(pid).cmdline()
        if any(x in cmd for x in ('train.agent_b.round8.campaign','train.agent_b.round8.finish')):
            raise RuntimeError(f'B writer already running at PID {pid}')
    assert 'B2-30-300' in s['confirmed']
    publish(p,h,s)
    queue=[(a,n,c) for a in ('B3-shared','B3-local-low') for n in (1,2) for c in DEV]
    end=datetime.fromisoformat(p['deadlines']['confirmation_end_utc'])
    s.update(process_id=os.getpid(),queue=queue,next_action='Confirm B3 against shared and intact controls')
    write(path,s)
    for a,n,c in queue:
        if c in s['results'].get(a,{}).get(str(n),{}): continue
        if datetime.now(timezone.utc)>=end or Path(p['integration']['freeze']).exists(): break
        key=f'{a}/s{n}/{c}'; s['active']=key; write(path,s)
        try:
            result=run(a,c,n)
            s['results'].setdefault(a,{}).setdefault(str(n),{})[c]=result
            print(key,result['evaluation']['candidate']['official'],flush=True)
        except Pending as exc:
            s.setdefault('pending',{})[key]=str(exc)
        finally:
            s['active']=None; write(path,s)
    s.update(next_action='Audit and hand off to A',queue=[]); publish(p,h,s)
    audit()
    s.update(process_id=None,next_action='A integration only; no new searches',active=None)
    publish(p,h,s,done=True)


if __name__=='__main__': main()
