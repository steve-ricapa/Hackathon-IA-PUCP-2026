"""Fold-local EOF bases and regularized prediction of their coefficients."""
from dataclasses import dataclass

import numpy as np
import torch

from train.shared.geo import bilinear

COUNTS = (16, 32, 64, 128)
GROUPS = {'separate': ((0,), (1,), (2,), (3,)), 'joint-uv': ((0,), (1,), (2,3))}
CONFIGS = [(g,k,reg) for g in GROUPS for k in COUNTS for reg in ('ridge','rrr')]


def features(windows, hours, missing, static):
    """Retain every GFS cell at each lag; add local WRF footprint averages."""
    n,s = windows.shape[:2]
    local = np.empty((n,s,2,2,4), np.float32)
    for i in range(0,n,32):
        for j in range(s):
            fine = bilinear(windows[i:i+32,j],static)
            for r, rows in enumerate(np.array_split(np.arange(fine.shape[1]),2)):
                for c, cols in enumerate(np.array_split(np.arange(fine.shape[2]),2)):
                    local[i:i+len(fine),j,r,c] = fine[:,rows][:,:,cols].mean(axis=(1,2))
    angle = hours * (np.pi/12)
    return np.column_stack((windows.reshape(n,-1),local.reshape(n,-1),missing,
                            np.sin(angle),np.cos(angle))).astype(np.float32)


@dataclass
class Basis:
    channels: tuple
    mean: torch.Tensor
    vectors: torch.Tensor
    ratios: torch.Tensor

    def project(self,y,k):
        return (y[...,list(self.channels)].reshape(len(y),-1)-self.mean) @ self.vectors[:,:k]

    def expand(self,z):
        return z @ self.vectors[:,:z.shape[1]].T + self.mean


def basis(y,channels,seed,max_k=128):
    """Randomized EOF with oversampling, fitted only to passed training rows."""
    z=y[...,list(channels)].reshape(len(y),-1)
    mean=z.mean(0)
    z=z-mean
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        _, s, v=torch.pca_lowrank(z,q=min(max_k+24,*z.shape),center=False,niter=4)
    var=z.square().sum().clamp_min(1e-12)
    vectors=v[:,:max_k].contiguous()
    # Compute captured variance directly, rather than using approximate singular values.
    ratios=(z@vectors).square().sum(0)/var
    return Basis(channels,mean,vectors,ratios)


class Design:
    def __init__(self,x):
        x=torch.as_tensor(x)
        self.mean=x.mean(0)
        self.scale=x.std(0,unbiased=False).clamp_min(1e-5)
        self.x=self.transform(x)
        # Ridge penalty fixed a priori: 0.1 × n in standardized coordinates.
        gram=self.x@self.x.T
        gram.diagonal().add_(0.1*len(x))
        self.chol=torch.linalg.cholesky(gram)

    def transform(self,x):
        return (torch.as_tensor(x)-self.mean)/self.scale

    def fit(self,z,reg):
        mean=z.mean(0)
        centered=z-mean
        w=self.x.T@torch.cholesky_solve(centered,self.chol)
        if reg=='rrr':
            # Ridge reduced-rank objective uses eigenspace of Z'X(X'X+λI)^−1X'Z.
            fitted=self.x@w
            cov=centered.T@fitted
            _,v=torch.linalg.eigh((cov+cov.T)*0.5)
            v=v[:,-max(4,z.shape[1]//2):]
            w=(w@v)@v.T
        elif reg!='ridge':
            raise ValueError(reg)
        return w,mean


def predict(bases,design,x,k,reg,y_train,output_shape):
    result=torch.empty((len(x),*output_shape),dtype=torch.float32)
    models=[]
    vx=design.transform(x)
    for b in bases:
        w,intercept=design.fit(b.project(y_train,k),reg)
        result[...,list(b.channels)]=b.expand(vx@w+intercept).reshape(len(x),*output_shape[:-1],len(b.channels))
        models.append(dict(channels=list(b.channels),mean=b.mean,vectors=b.vectors[:,:k].contiguous(),
                           weights=w,intercept=intercept))
    return result,models


def oracle(bases,y,k):
    """Diagnostics only. Never use this result for ranking predictive candidates."""
    result=torch.empty_like(y)
    for b in bases:
        result[...,list(b.channels)]=b.expand(b.project(y,k)).reshape(*y.shape[:-1],len(b.channels))
    return result
