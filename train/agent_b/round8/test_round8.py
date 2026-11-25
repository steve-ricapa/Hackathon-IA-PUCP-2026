import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from .control import starter
from .mechanisms import batches,weights
from .run import ARMS,model_for
from .campaign import paired
from train.agent_b.round4.run import pool


class Checks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_weights_and_sampling(self):
        t=pd.date_range('2024-01-01',periods=80,freq='D',tz='UTC'); end=t[-1]+pd.Timedelta(days=1)
        w,d=weights(t,end,60)
        self.assertAlmostEqual(float(w.mean()),1.,places=6)
        self.assertAlmostEqual(float(w[60]/w[0]),2.,places=6)
        self.assertLess(d['effective_hours'],80)
        b=batches(len(t),0); self.assertEqual(set(np.concatenate([next(b) for _ in range(3)])),set(range(80)))
        b=batches(len(t),0,times=t,cutoff=end,days=30); self.assertEqual(set(next(b)),set(range(50,80)))

    def test_starter_scope_and_identity(self):
        static=dict(dem=np.arange(2450,dtype='float32').reshape(70,35),gfs_lats=np.arange(17,dtype='float32')[::-1],
                    gfs_lons=np.arange(13,dtype='float32'),wrf_lats=np.linspace(0,16,70,dtype='float32'),
                    wrf_lons=np.linspace(0,12,35,dtype='float32'))
        d=SimpleNamespace(static=static); base=starter(static); ck={'state_dict':base.state_dict()}
        x=torch.randn(2,3,17,13,4); h=torch.tensor([0,12]); m=torch.zeros(2,3,dtype=torch.bool)
        for name in ('B2-30-100','B3-shared','B3-local-low','B3-local-high'):
            mod=model_for(d,ARMS[name],ck); p=mod(x,h,m)
            torch.testing.assert_close(base(x,h,m),p,atol=0,rtol=0)
            p.square().mean().backward()
            for k,v in mod.named_parameters():
                if name.startswith('B2'): self.assertEqual(v.requires_grad,k.startswith('dec.'))
                elif k.startswith('base.'): self.assertIsNone(v.grad)
            if name.startswith('B3'):
                self.assertEqual(sum(v.numel() for v in mod.parameters() if v.requires_grad),272+(39200 if mod.delta is not None else 0))

    def test_selection_rejects_one_week_gain(self):
        def m(x): return pool([dict(rmse=[x]*4,hours=8)])
        a=[dict(candidate=m(.9),control=m(1.),weeks=[[m(.9),m(1.)],[m(.9),m(1.)]]) for _ in range(3)]
        self.assertTrue(paired(a)['promoted'])
        for e in a: e['weeks']=[[m(1.1),m(1.)],[m(1.1),m(1.)]]
        a[0]['weeks'][0][0]=m(.1)
        self.assertFalse(paired(a)['promoted'])


if __name__=='__main__': unittest.main()
