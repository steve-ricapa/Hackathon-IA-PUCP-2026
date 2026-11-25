import unittest

import numpy as np
import torch

from .models import kernel
from .run import SPECS, protocol


class Round7Tests(unittest.TestCase):
    def test_kernel_gram_positive_with_regularization(self):
        g=torch.Generator().manual_seed(12)
        x=torch.randn((20,61),generator=g)/61**.5
        t=torch.arange(20)*.2; h=torch.stack([t.sin(),t.cos()],1)
        y=torch.randn((20,25),generator=g)
        for kind in ('linear','rbf','clock'):
            gram=kernel(x,x,h,h,kind,.45)
            torch.testing.assert_close(gram,gram.T,atol=1e-6,rtol=0)
            gram.diagonal().add_(.05)
            l=torch.linalg.cholesky(gram)
            w=torch.cat([torch.cholesky_solve(q.contiguous(),l) for q in y.split(7,1)],1)
            torch.testing.assert_close(gram@w,y,atol=5e-5,rtol=1e-5)

    def test_protocol_and_budget(self):
        p,_,origin=protocol()
        self.assertEqual(origin,'round7')
        self.assertEqual(p['development'],['C0','C1','C2'])
        self.assertLessEqual(sum(s is not None for s in SPECS.values()),p['budget']['max_configurations'])
        self.assertLessEqual(len(SPECS),8)


if __name__=='__main__': unittest.main()
