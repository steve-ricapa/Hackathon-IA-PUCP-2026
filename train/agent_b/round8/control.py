"""Read-only provenance gate and starter imports without changing shared files."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from train.agent_b.round4.run import ORIG, read, sha

ROUND=ORIG/'.private/experiments/round8'
ROOT=ROUND/'agent_b'


class Pending(RuntimeError): pass


def source(name,path):
    spec=spec_from_file_location(name,path)
    mod=module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def starter(static):
    m=source('train.shared.r8_original_starter',ORIG/'train/shared/starter.py').StarterCNN(static)
    if sum(p.numel() for p in m.parameters())!=143956:
        raise ValueError('Starter parameter count mismatch')
    return m


def protocol():
    p=ROUND/'protocol.json'; ready=ROUND/'READY'
    if not p.is_file() or not ready.is_file(): raise Pending('Waiting for A protocol.json and READY')
    h=sha(p); marker=ready.read_text(encoding='utf-8').strip()
    if marker!=h: raise ValueError('READY does not match protocol SHA-256')
    data=read(p)
    if tuple(data['development'])!=('C0','C1','C2'): raise ValueError('Unexpected development folds')
    if Path(data['paths']['gpu_lock']).resolve()!=(ORIG/'.private/gpu.lock').resolve():
        raise ValueError('Expected original shared GPU lock')
    return data,h


def artifact(node,home):
    p=(home/node['path']).resolve()
    if not p.is_file(): raise Pending(f'Pending {p.name}')
    if sha(p)!=node['sha256']: raise ValueError(f'Hash mismatch: {p}')
    return p


def verify_recipe(cfg):
    exact=dict(epochs=25,batch=32,lr=.002,weight_decay=.0001)
    if any(cfg.get(k)!=v for k,v in exact.items()) or tuple(cfg['lags'])!=(2,1,0):
        raise ValueError(f'Starter recipe mismatch: {cfg}')
