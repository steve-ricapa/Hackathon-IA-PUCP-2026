import math

import numpy as np
import torch

from train.agent_b.round3.eof import basis, Design, features
from train.shared.geo import bilinear


def eof_wind(xtr, xva, ytr, hours_tr, hours_va, spec):
    y = torch.from_numpy(ytr)
    b = basis(y, (2, 3), 0, spec['modes'])
    z = b.project(y, spec['modes'])
    if spec['kind'] == 'kernel':
        d = Design(xtr)
        a, v = d.x, d.transform(xva)
        def kernel(p, q):
            return (p @ q.T)/p.shape[1] + torch.exp(-torch.cdist(p,q).square()/(2*p.shape[1]))
        gram = kernel(a,a)
        gram.diagonal().add_(spec['ridge'])
        mean = z.mean(0)
        dual = torch.cholesky_solve(z-mean,torch.linalg.cholesky(gram))
        coef = kernel(v,a)@dual+mean
        fit = dict(weights=dual, anchors=a, intercept=mean, mean=d.mean, scale=d.scale)
    else:
        d=Design(xtr)
        w, intercept=d.fit(z,'rrr')
        coef=d.transform(xva)@w+intercept
        fit=dict(weights=w,intercept=intercept,mean=d.mean,scale=d.scale)
    return b.expand(coef).reshape(len(xva),*y.shape[1:3],2).numpy(), dict(fit=fit,basis=b.vectors,center=b.mean)


def local_design(w, h, static, kind):
    f = [bilinear(w[:,k],static) for k in range(w.shape[1])]
    current=f[-1]; u,v=current[...,2],current[...,3]
    parts = [current, f[0][...,2:], f[1][...,2:]]
    angle = h[:,None,None]*(math.pi/12)
    ones = np.ones_like(u)
    parts.append(np.stack((ones*np.sin(angle),ones*np.cos(angle), np.hypot(u,v)),axis=-1))
    if kind=='terrain':
        dem=static['dem']; dy,dx=np.gradient(dem)
        slopes=np.maximum(20,np.hypot(dx,dy)); nx,ny=dx/slopes,dy/slopes
        along=u*nx+v*ny; across=-u*ny+v*nx
        parts.append(np.stack((along,across, along*np.hypot(u,v), across*np.hypot(u,v),
                               along*np.sin(angle),across*np.cos(angle)),axis=-1))
    if kind=='dynamics':
        grads=np.gradient(current[...,2:],axis=(1,2))
        change=current[...,2:]-f[0][...,2:]
        parts.extend([*grads,change,change*np.hypot(u,v)[...,None]])
    return torch.from_numpy(np.concatenate(parts,axis=-1).astype(np.float32)).reshape(len(w),-1,sum(p.shape[-1] for p in parts)).transpose(0,1)


def local_regression(a,v,y,ridge):
    """Independent regularized local fit for each fixed pixel, in chunks."""
    a,v=a.contiguous(),v.contiguous()
    y=torch.from_numpy(y).reshape(len(y),-1,2).transpose(0,1)
    result=[]; weights=[]; means=[]; scales=[]; targets=[]
    for p in range(0,len(a),128):
        x, q, z = a[p:p+128],v[p:p+128],y[p:p+128]
        mu=x.mean(1,keepdim=True); sd=x.std(1,keepdim=True,unbiased=False).clamp_min(1e-4)
        ym=z.mean(1,keepdim=True); xs=(x-mu)/sd
        g=xs.transpose(1,2)@xs
        g.diagonal(dim1=-2,dim2=-1).add_(ridge*x.shape[1])
        w=torch.cholesky_solve(xs.transpose(1,2)@(z-ym),torch.linalg.cholesky(g))
        result.append((q-mu)/sd@w+ym);weights.append(w);means.append(mu);scales.append(sd);targets.append(ym)
    return torch.cat(result).transpose(0,1).contiguous().numpy(),dict(weights=torch.cat(weights),
            mean=torch.cat(means),scale=torch.cat(scales),intercept=torch.cat(targets))
