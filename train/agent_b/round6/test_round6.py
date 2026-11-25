import unittest

import numpy as np
import torch

from .model import WindNet,loss
from .physics import features
from .run import protocol


class WindChecks(unittest.TestCase):
    def test_protocol(self):
        self.assertEqual(protocol()[0]['development'],['C0','C1','C2'])

    def test_upstream_period_and_calms(self):
        lat=np.arange(30,dtype='float32')*.035-10
        lon=np.arange(30,dtype='float32')*.035-72
        dem=np.broadcast_to(np.arange(30,dtype='float32')*50,(30,30)).copy()
        static=dict(wrf_lats=lat,wrf_lons=lon,dem=dem)
        x=np.zeros((3,30,30,4),dtype='float32'); x[0,...,2]=2; x[1,...,2]=-2
        z,p=features(static,x)
        self.assertGreater(float(z[1].mean()),0.)
        self.assertLess(float(p[0,11,15,15]),0.)
        self.assertGreater(float(p[1,11,15,15]),0.)
        self.assertEqual(float(p[2,11:].abs().max()),0.)

    def test_heads_finite_at_calm(self):
        for k in ('uv','polar','hybrid','angle'):
            y,polar=WindNet(12,8,k,'local')(torch.zeros(2,12,12,8))
            self.assertEqual(y.shape,(2,2,12,8))
            if polar is not None:
                speed,unit=polar
                self.assertTrue(torch.all(speed>=0))
                torch.testing.assert_close(unit.norm(dim=1),torch.ones_like(speed[:,0]))
            obj=loss((y,polar),torch.zeros_like(y),torch.tensor([.45,.55]),aux=.10)
            self.assertTrue(torch.isfinite(obj)); obj.backward()


if __name__=='__main__': unittest.main()
