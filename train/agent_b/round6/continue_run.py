"""Continue matched screens and confirmation after the first wave finishes."""
import subprocess
import sys
import time

from train.agent_b.round4.run import read, write
from .run import ROOT, SPECS, protocol, ranking


def main():
    proto,_=protocol()
    if not all(f'{k}-s0-e5' in read(ROOT/'research_state.json')['experiments']
               for k in ('uv','polar','hybrid','physics')):
        raise RuntimeError('Finish the first wave before starting this controller')

    def state(): return read(ROOT/'research_state.json')

    def available():
        s=state()
        return (time.time()-s['started']<min(180,proto['budget']['minutes'])*60 and
                len(s['experiments'])<min(60,proto['budget']['max_configurations']))

    def batch(names,epochs=5,seed=0):
        names=[k for k in names if f'{k}-s{seed}-e{epochs}' not in state()['experiments']]
        if names and available():
            subprocess.run([sys.executable,'-B','-u','-m','train.agent_b.round6.run',
                            '--epochs',str(epochs),'--seed',str(seed),*names],check=True)

    def top(epochs,limit=2):
        return [r[0].rsplit('-s',1)[0] for r in ranking(state()) if r[0].endswith(f'-s0-e{epochs}')][:limit]

    batch(('diurnal-control','diurnal-physics','pyramid-control','pyramid-physics'))
    batch(('polar-physics','hybrid-physics','polar-aux'))
    batch(('spectral','extreme','angle'))
    # Confirm at fixed capacity, even if the result is negative: compare the best
    # physics architecture to its matched control (plus the best head if distinct).
    finalists=list(dict.fromkeys(top(5)+['uv','physics']))
    for seed in (1,2): batch(finalists,seed=seed)
    # Larger epoch cap is a check for underfitting, not a change to the inner rule.
    for epochs in (9,14):
        batch(list(dict.fromkeys([*top(5,4),'uv','physics'])),epochs=epochs)
        for seed in (1,2): batch(top(epochs),epochs=epochs,seed=seed)
    for seed in (1,2): batch(SPECS,seed=seed)
    if available(): batch(SPECS,epochs=9)
    if available():
        raise RuntimeError('Planned work exhausted before the budget; expand structural comparisons')
    s=state(); s['stop_reason']=('configuration_budget' if len(s['experiments'])>=proto['budget']['max_configurations'] else 'time_budget')
    write(ROOT/'research_state.json',s)
    subprocess.run([sys.executable,'-B','-m','train.agent_b.round6.report','--publish'],check=True)


if __name__=='__main__': main()
