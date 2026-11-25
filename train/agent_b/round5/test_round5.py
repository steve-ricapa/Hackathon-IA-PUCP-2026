import unittest

import numpy as np

from .loop import Development, INCUMBENT, nested, ranking
from .regional import weights
from .spectral import dct


def result(a,b):
    def metric(x):
        return dict(official=x,rmse=[1-x]*4,hours=8)
    return dict(score=metric((a+b)/2),slices={'(7, 0)':metric(a),'(7, 1)':metric(b)})


class BenchmarkTests(unittest.TestCase):
    def test_dct_roundtrip(self):
        x=np.random.default_rng(2).normal(size=(3,5,4,2)).astype('float32')
        np.testing.assert_allclose(dct(dct(x,5,4),5,4,inverse=True,shape=(5,4)),x,atol=1e-6)

    def test_regional_partition(self):
        for g in ((1,1),(2,1),(4,3)):
            np.testing.assert_allclose(weights((70,35),g).sum(0),1,atol=1e-6)

    def test_no_c3(self):
        with self.assertRaises(ValueError):
            object.__new__(Development).fold('C3')

    def test_selection_cannot_consult_outer(self):
        cuts={f'C{i}':result(.5,.5) for i in range(3)}
        state=dict(incumbent=dict(cuts=cuts,pooled=dict(official=.5)),experiments={
            'late-only': dict(cuts={**cuts,'C2':result(.9,.9)},pooled=dict(official=.6))})
        self.assertEqual(nested(state)['C2']['selected'],INCUMBENT)
        self.assertEqual(ranking(state)[0],INCUMBENT)


if __name__=='__main__':
    unittest.main()
