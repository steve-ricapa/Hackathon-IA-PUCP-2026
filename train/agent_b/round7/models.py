"""Full-field shared-kernel regression and spatial MOS on CPU."""
import numpy as np
import torch
from torch.nn import functional as F

from train.shared.geo import bilinear
from train.shared.preprocess import temporal_windows


def climatology(x,tr,h):
    return np.stack([x[tr[h[tr]==j]].mean(0) for j in range(24)])


def regional(ds,tr,va,lags,anomaly):
    stop=va[-1]+1; times=ds.times[:stop]; h=times.hour.to_numpy(); x=ds.x[:stop]
    climate=climatology(x,tr,h)
    raw=x-climate[h] if anomaly else x.copy()
    scale=raw[tr].std((0,1,2)).clip(.01)
    fields,missing=temporal_windows(raw/scale,times,lags)
    a=torch.from_numpy(fields.reshape(stop,-1)); a/=a.shape[1]**.5
    a=torch.cat([a,torch.from_numpy(missing.astype('float32'))*.10],dim=1)
    angle=torch.from_numpy((h*np.pi/12).astype('float32'))
    hour=torch.stack([angle.sin(),angle.cos()],dim=1)
    return a,hour,dict(gfs_climate=torch.from_numpy(climate),scale=torch.from_numpy(scale))


def kernel(a,b,h,g,kind,width):
    clock=h@g.T
    linear=a@b.T
    if kind=='linear': return linear+.15*clock+1
    dist=(a.square().sum(1)[:,None]+b.square().sum(1)[None,:]-2*linear).clamp_min_(0)
    rbf=torch.exp(-dist/(2*width))
    if kind=='clock': rbf.mul_(.5+.5*torch.exp(clock-1))
    return rbf+.15*clock+1


def krr(ds,tr,va,spec,params):
    kind,lags,anomaly=spec; a,h,state=regional(ds,tr,va,lags,anomaly)
    hours=ds.times.hour.to_numpy(); clim=climatology(ds.y,tr,hours)
    target=torch.from_numpy((ds.y[tr]-clim[hours[tr]]).reshape(len(tr),-1))
    suppress_checkpoint=len(params)>1
    for width,penalty in params:
        gram=kernel(a[tr],a[tr],h[tr],h[tr],kind,width)
        gram.diagonal().add_(penalty)
        chol=torch.linalg.cholesky(gram); del gram
        query=kernel(a[va],a[tr],h[va],h[tr],kind,width)
        pred=torch.empty((len(va),target.shape[1])); dual=None if suppress_checkpoint else torch.empty_like(target)
        # One factorization serves RH/T/U/V and every pixel; no output basis.
        for j in range(0,target.shape[1],320):
            sl=slice(j,j+320); w=torch.cholesky_solve(target[:,sl].contiguous(),chol)
            pred[:,sl]=query@w
            if dual is not None: dual[:,sl]=w
        ck={} if dual is None else dict(**state,dual=dual,anchors=a[tr],hours=h[tr],
                 climate=torch.from_numpy(clim),spec=list(spec),params=[width,penalty])
        yield (width,penalty),pred.numpy().reshape(len(va),*clim.shape[1:])+clim[hours[va]],ck


def mos_fields(ds,stop):
    x=ds.x[:stop]; w,m=temporal_windows(x,ds.times[:stop],(2,0))
    # Symmetric three-cell GFS neighborhood before geographic interpolation.
    coarse=torch.from_numpy(x).permute(0,3,1,2)
    neighbors=F.avg_pool2d(F.pad(coarse,(1,1,1,1),mode='replicate'),3,stride=1).permute(0,2,3,1).numpy()
    fields=[bilinear(a,ds.static).reshape(stop,-1,4) for a in (x,neighbors,w[:,0])]
    return fields,m[:,0]


def mos(ds,tr,va,spec,params,fields):
    (local,near,past),missing=fields
    h=ds.times.hour.to_numpy(); i=np.r_[tr,va]; shape=ds.y.shape[1:]; n=len(tr)
    clim=climatology(ds.y,tr,h).reshape(24,-1,4); ref=clim[h[i]]
    gc=climatology(local,tr,h); nc=climatology(near,tr,h)
    ys=(ds.y[tr].reshape(n,-1,4)-ref[:n]); dem=ds.static['dem'].reshape(-1)
    elevation=(dem-dem.mean())/max(1.,dem.std())
    preds=[np.empty((len(va),len(dem),4),dtype='float32') for _ in params]; ck=[]
    for j in range(0,len(dem),80):
        sl=slice(j,j+80); a=local[i,sl]-gc[h[i],sl]; b=near[i,sl]-nc[h[i],sl]
        step=local[i,sl]-past[i,sl]; step[missing[i]]=0
        p=a.shape[1]; ang=h[i,None,None]*np.pi/12
        cyc=np.broadcast_to(np.concatenate([np.sin(ang),np.cos(ang)],axis=2),(len(i),p,2))
        inputs=[a,b,step,ref[:,sl],cyc]
        if spec=='nonlinear':
            inputs.extend([a*np.abs(a),a*cyc[...,:1],a*cyc[...,1:],a*ref[:,sl],
                           np.hypot(local[i,sl,2],local[i,sl,3])[...,None]])
        x=torch.from_numpy(np.concatenate(inputs,axis=2).astype('float32')).transpose(0,1).contiguous()
        z=torch.from_numpy(ys[:,sl]).transpose(0,1); bias=z.mean(1,keepdim=True)
        mu=x[:,:n].mean(1,keepdim=True); sd=x[:,:n].std(1,keepdim=True,unbiased=False).clamp_min(.01)
        x=(x-mu)/sd; t=x[:,:n]; g=t.transpose(1,2)@t; rhs=t.transpose(1,2)@(z-bias)
        # Elevation controls pooling strength; a constant DEM column is not identifiable in local regression.
        shrink=torch.from_numpy((1+.20*np.abs(elevation[sl])).astype('float32'))[:,None]
        for k,penalty in enumerate(params):
            gram=g.clone(); gram.diagonal(dim1=-2,dim2=-1).add_(penalty*n*shrink)
            beta=torch.cholesky_solve(rhs,torch.linalg.cholesky(gram))
            preds[k][:,sl]=(x[:,n:]@beta+bias).transpose(0,1).numpy()+ref[n:,sl]
            if len(params)==1: ck.append(dict(beta=beta,mean=mu,sd=sd,bias=bias))
    for param,pred in zip(params,preds):
        state={} if len(params)>1 else dict(chunks=ck,climate=torch.from_numpy(clim),
            gfs_climate=torch.from_numpy(gc),neighbor_climate=torch.from_numpy(nc),spec=spec,penalty=param)
        yield param,pred.reshape(len(va),*shape),state
