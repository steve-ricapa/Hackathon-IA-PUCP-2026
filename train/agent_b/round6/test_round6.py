import unittest

import numpy as np
import pandas as pd
import torch

from .model import WindNet,loss
from .physics import features
from .run import protocol
from .temporal import encode, transport


class WindChecks(unittest.TestCase):
    def test_protocol(self):
        self.assertEqual(protocol()[0]['development'],['C0','C1','C2'])

    def test_temporal_rejects_future_and_ignores_later_frames(self):
        times=pd.date_range('2024-01-01',periods=72,freq='h',tz='UTC').delete(27)
        x=np.random.default_rng(0).normal(size=(len(times),8,8,4)).astype('float32')
        tr=np.arange(24)
        with self.assertRaises(ValueError): encode(x,times,tr,'absolute',(1,0,-1))
        for mode in ('absolute','anomaly','tendency'):
            a,_=encode(x,times,tr,mode,(6,3,0))
            x2=x.copy(); x2[40:]+=1000
            b,_=encode(x2,times,tr,mode,(6,3,0))
            torch.testing.assert_close(a[:40],b[:40],rtol=0,atol=0)

    def test_eastward_advection_uses_western_values(self):
        lat=np.arange(4,dtype='float32')[::-1]*.25
        lon=np.arange(6,dtype='float32')*.25
        w=np.zeros((1,2,4,6,4),dtype='float32'); w[0,0,:,:,0]=np.arange(6)
        flow=np.zeros_like(w)
        flow[0,0,:,:,2]=np.deg2rad(.25)*6371000*np.cos(np.deg2rad(lat))[:,None]/3600
        p,missing=transport(w,np.zeros((1,2),dtype=bool),(1,0),dict(gfs_lats=lat,gfs_lons=lon),flow)
        np.testing.assert_allclose(p[0,0,:,1:,0],w[0,0,:,:-1,0],atol=1e-5)
        self.assertFalse(missing.any())

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
