"""Small spatial-basis regressions, not neural operators."""
import numpy as np
import torch
from functools import lru_cache

from train.shared.preprocess import temporal_windows


@lru_cache
def cosine(n,k):
    basis=np.cos(np.pi/n*np.arange(k)[:,None]*(np.arange(n)+.5))*np.sqrt(2/n)
    basis[0]/=np.sqrt(2)
    return torch.tensor(basis,dtype=torch.float32)


def dct(x,rows,cols,inverse=False,shape=None):
    nr,nc=shape if inverse else x.shape[-3:-1]
    a,b=cosine(nr,rows),cosine(nc,cols)
    if inverse:
        a,b=a.T,b.T
    t=torch.from_numpy(np.ascontiguousarray(x)).movedim(-1,-3)
    return (a@t@b.T).movedim(-3,-1).numpy()


def encode(x, transform, k):
    if transform == 'dct':
        return dct(x,k,k)
    # One Haar-style decomposition with replicated boundary samples.
    y = np.pad(x, [(0,0)]*(x.ndim-3)+[(0,x.shape[-3]%2),(0,x.shape[-2]%2),(0,0)], mode='edge')
    a,b,c,d = (y[...,i::2,j::2,:] for i,j in ((0,0),(0,1),(1,0),(1,1)))
    low=(a+b+c+d)/2
    if k == 1:
        return low
    return np.concatenate([low,(a+b-c-d)/2,(a-b+c-d)/2,(a-b-c+d)/2],axis=-1)


def design(data, tr, va, n, spec):
    ds=data.ds; stop=va[-1]+1
    w,m=temporal_windows(ds.x[:stop],ds.times[:stop],spec['lags'])
    h=ds.times.hour.to_numpy()[:stop]*np.pi/12
    x=encode(n.inputs(w),spec['transform'],spec['k']).reshape(stop,-1)
    if spec.get('quadratic'):
        x=np.column_stack([x,np.square(x)])
    x=torch.from_numpy(np.column_stack([x,m,np.sin(h),np.cos(h)]).astype('float32'))
    xtr=x[tr]; mean=xtr.mean(0); sd=xtr.std(0,unbiased=False).clamp_min(1e-4)
    return (xtr-mean)/sd,(x[va]-mean)/sd,mean,sd


def fit(data, cut, spec):
    ds=data.ds
    tr,va,h,_,_,_,n,c=data.fold(cut)
    a,v,mean,sd=design(data,tr,va,n,spec)
    rows,cols=spec['output']; y=n.targets(ds.y[tr],h[tr])
    modes=dct(y,rows,cols)
    z=torch.from_numpy(modes.reshape(len(tr),-1)); intercept=z.mean(0)
    x=a.T@a; x.diagonal().add_(spec['ridge']*len(tr))
    coef=torch.cholesky_solve(a.T@(z-intercept),torch.linalg.cholesky(x))
    low=(v@coef+intercept).numpy().reshape(len(va),rows,cols,4)
    pred=dct(low,rows,cols,inverse=True,shape=y.shape[1:3])*n.anom_sd+c
    return pred,dict(coef=coef,intercept=intercept,mean=mean,sd=sd)


def configurations():
    variants=[('dct',4,(10,7),.10,(0,),False),('dct',6,(20,12),.10,(0,),False),
              ('dct',6,(20,12),.15,(2,1,0),False),('dct',4,(20,12),.15,(6,2,0),False),
              ('dct',4,(20,12),.15,(2,1,0),True),('dct',8,(24,16),.40,(2,1,0),False),
              ('haar',1,(20,12),.20,(2,1,0),False),('haar',2,(20,12),.40,(0,),False)]
    for i,(method,k,output,ridge,lags,q) in enumerate(variants,1):
        yield f'spectral-{i:02}',dict(transform=method,k=k,output=output,ridge=ridge,lags=lags,quadratic=q)
