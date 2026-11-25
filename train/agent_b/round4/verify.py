"""Verify development artifacts and freeze a selection without touching C3."""
import argparse

import numpy as np
import torch

from train.shared.artifacts import aligned_predictions
from train.shared.metrics import score_mask
from .run import DEV, ROOT, R3, Data, blocks, frozen_config, measure, pool, read, sha, write


def verify(data,cut,entry):
    _,_,_,t,bounds,base,n,_=data.fold(cut)
    for key in ('predictions','checkpoint','normalizer'):
        art=entry[key]; assert sha(ROOT/art['path'])==art['sha256']
    meta,(pred,)=aligned_predictions([ROOT/entry['predictions']['path']])
    np.testing.assert_array_equal(meta['times'],t.strftime('%Y%m%d%H'))
    np.testing.assert_array_equal(meta['scored'],score_mask(t))
    assert (meta['start'].item(),meta['end_exclusive'].item())==bounds
    np.testing.assert_array_equal(meta['wrf_lats'],data.ds.static['wrf_lats'])
    np.testing.assert_array_equal(meta['wrf_lons'],data.ds.static['wrf_lons'])
    np.testing.assert_array_equal(pred[...,:2],base[...,:2])
    assert measure(data,cut,pred)==entry['metrics'] and blocks(data,cut,pred)==entry['blocks']
    torch.load(ROOT/entry['checkpoint']['path'],map_location='cpu',weights_only=True)
    with np.load(ROOT/entry['normalizer']['path'],allow_pickle=False) as file:
        for k in vars(n):np.testing.assert_array_equal(file[k],getattr(n,k))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','audit-final'));a=p.parse_args()
    data=Data()
    if a.stage=='audit-final':
        freeze=frozen_config(); entry=read(ROOT/'final.json')
        folder=ROOT/'runs'/freeze['name']/'C3'
        assert entry==read(folder/'summary.json')
        for key,name in [('predictions','predictions.npz'),('checkpoint','best.pt'),('normalizer','normalizer.npz')]:
            assert ROOT/entry[key]['path']==folder/name
        verify(data,'C3',entry)
        assert measure(data,'C3',data.fold('C3')[5])==entry['reference']
        np.testing.assert_allclose(entry['reference']['official'],
                                   read(R3/'integration/report.json')['chosen_C3']['official'],atol=1e-8,rtol=0)
        print('C3 verified');return
    dest=ROOT/'best_config.json'
    if dest.exists():raise FileExistsError(dest)
    state=read(ROOT/'research_state.json'); baseline=pool(list(state['baselines'].values()))
    for n,v in state['experiments'].items():
        for c in DEV:verify(data,c,v['cuts'][c])
        assert pool([v['cuts'][c]['metrics'] for c in DEV])==v['pooled']
    eligible=[(n,v) for n,v in state['experiments'].items() if all(v['cuts'][c]['metrics']['official']>=state['baselines'][c]['official'] for c in DEV)]
    name,v=max(eligible,key=lambda x:x[1]['pooled']['official'],default=('reference',dict(pooled=baseline)))
    if v['pooled']['official']<=baseline['official']:name,v='reference',dict(pooled=baseline)
    write(dest,dict(name=name,**{k:v[k] for k in ('config','weight','pooled') if k in v},baseline=baseline,
                    selection='Highest pooled C0/C1/C2 with no cut worse than baseline. Never look at C3.',
                    protocol_sha256=sha(R3/'protocol.json'), reference_sha256=sha(R3/'integration/frozen.json'),
                    development_sha256=sha(ROOT/'research_state.json')))
    print('Verified',len(state['experiments'])*len(DEV),'development artifacts; frozen:',name,v['pooled'])


if __name__=='__main__':main()
