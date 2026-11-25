"""Regional analogs: train-only neighbors of causal GFS weather states."""
import numpy as np
import torch

from train.shared.geo import bilinear
from train.shared.preprocess import temporal_windows


def weights(shape,grid):
    axes=[]
    for n,k in zip(shape,grid):
        domain=np.linspace(0,1,n); centers=np.linspace(0,1,k)
        a=np.maximum(0,1-np.abs(domain[None,:]-centers[:,None])*max(1,k-1)) if k>1 else np.ones((1,n))
        axes.append(a)
    a,b=axes
    return (a[:,None,:,None]*b[None,:,None,:]).reshape(-1,*shape).astype('float32')


def fit(data,cut,spec):
    ds=data.ds
    tr,va,h,t,_,_,n,c=data.fold(cut)
    stop=va[-1]+1; lags=spec['lags']
    x,m=temporal_windows(ds.x[:stop],ds.times[:stop],lags)
    x=n.inputs(x)
    y=torch.from_numpy(n.targets(ds.y[tr],h[tr]).reshape(len(tr),-1,4))
    fine=np.stack([bilinear(x[:,j],ds.static) for j in range(len(lags))],axis=1)
    dem=ds.static['dem']; dy,dx=np.gradient(dem)
    lat=ds.static['wrf_lats']; lon=ds.static['wrf_lons']
    dx=dx/(np.cos(np.deg2rad(lat))[:,None]*111_200*np.mean(np.diff(lon)))
    dy=dy/(111_200*np.mean(np.diff(lat)))
    angle=h[:stop]*np.pi/12
    p=torch.zeros((len(va),y.shape[1],4)); members=[]; coefs=[]
    for reg in weights(dem.shape,spec['grid']):
        support=reg>0; cell_weight=reg[support]; norm=cell_weight/cell_weight.sum()
        w=fine[:,:,support]; avg=(w*norm[None,None,:,None]).sum(2)
        f=[avg.reshape(stop,-1), m, np.sin(angle)[:,None], np.cos(angle)[:,None]]
        if spec['texture']:
            f.append(np.sqrt(((w[:,-1]-avg[:,-1,None])**2*norm[None,:,None]).sum(1)))
        if spec['terrain']:
            u,v=w[:,-1,:,2],w[:,-1,:,3]
            # Use physical (not standardized) GFS directions and metric slopes.
            u=u*n.x_sd[2]+n.x_mean[2]; v=v*n.x_sd[3]+n.x_mean[3]
            along=u*dx[support]+v*dy[support]
            f.extend([(along*norm).sum(1,keepdims=True),(abs(along)*norm).sum(1,keepdims=True)])
        feat=torch.from_numpy(np.column_stack(f).astype('float32'))
        a,b=feat[tr],feat[va]; sd=a.std(0,unbiased=False).clamp_min(.1)
        dist=torch.cdist(b/sd,a/sd).square()/a.shape[-1]
        val,neighbors=dist.topk(spec['k'],largest=False)
        prob=torch.softmax(-(val-val[:,:1])/spec['temperature'],dim=1)
        ix=torch.from_numpy(np.flatnonzero(support))
        for batch in range(0,len(va),16):
            inds=neighbors[batch:batch+16]; q=prob[batch:batch+16]
            hist=y[inds[:,:,None],ix[None,None,:]]
            p[batch:batch+16,ix]+=(hist*q[:,:,None,None]).sum(1)*torch.from_numpy(cell_weight)[None,:,None]
        members.append(torch.as_tensor(tr)[neighbors]); coefs.append(prob)
    return p.numpy().reshape(c.shape)*n.anom_sd+c,dict(indices=torch.stack(members),weights=torch.stack(coefs),
            grid=torch.from_numpy(weights(dem.shape,spec['grid'])),train_climatology=torch.from_numpy(n.clim),
            scale=torch.from_numpy(n.anom_sd),evaluation_dates=[str(x) for x in t])


def configurations():
    settings=[((1,1),32,.3,(0,),False,False),((3,2),32,.3,(0,),False,False),
              ((3,2),16,.2,(2,1,0),False,False),((3,2),48,.4,(6,2,0),True,False),
              ((4,3),48,.4,(2,1,0),False,True),((4,3),32,.3,(2,1,0),True,True),
              ((2,1),32,.3,(0,),True,True),((3,2),12,.1,(0,),True,False)]
    for i,(grid,k,temperature,lags,texture,terrain) in enumerate(settings,1):
        yield f'regional-{i:02}',dict(grid=grid,k=k,temperature=temperature,lags=lags,texture=texture,terrain=terrain)
