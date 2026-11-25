"""Metric local operators on the ascending-latitude WRF grid."""
import numpy as np
import torch
from torch.nn import functional as F

R = 6_371_000


def features(static, current):
    """Input is physical bilinearly interpolated RH/T/eastward/northward GFS."""
    z=torch.as_tensor(static['dem']); lat=np.deg2rad(static['wrf_lats'])
    dy=float(np.mean(np.diff(lat))*R)
    cos=torch.as_tensor(np.cos(lat))[:,None]
    dx=cos*float(np.deg2rad(np.mean(np.diff(static['wrf_lons'])))*R)
    u,v=torch.from_numpy(current[...,2:]).unbind(-1)
    speed=torch.hypot(u,v); east=u/speed.clamp_min(.5); north=v/speed.clamp_min(.5)
    def grad(a):
        return torch.gradient(a,dim=-1)[0]/dx,torch.gradient(a,dim=-2)[0]/dy
    sx,sy=grad(z); ux,uy=grad(u); vx,vy=grad(v); wx,wy=grad(speed)
    # Spherical metric terms; derivatives describe the coarse interpolated flow only.
    tan=torch.as_tensor(np.tan(lat))[:,None]/R
    fields=[speed,(ux+vy-v*tan)*1e4,(vx-uy+u*tan)*1e4,wx*1e4,wy*1e4]
    static_fields=[z/2000,sx*5,sy*5]
    for k in (5,13):
        p=k//2; sm=F.avg_pool2d(F.pad(z[None,None],(p,p,p,p),mode='replicate'),k,stride=1)[0,0]
        gx,gy=grad(sm); xx,xy=grad(gx); yx,yy=grad(gy); xy=(xy+yx)/2
        static_fields.extend([(z-sm)/300,(xx+yy)*5e4])
        # Hessian anisotropy along versus across the flow: a proxy, not a valley classifier.
        fields.extend([(gx*east+gy*north)*5,(-gx*north+gy*east)*5,
                       ((xx-yy)*(east.square()-north.square())+4*xy*east*north)*5e4])
    y,x=torch.meshgrid(torch.arange(z.shape[0]),torch.arange(z.shape[1]),indexing='ij')
    upstream=[]; valid=[]
    for distance in (12_000.,30_000.,60_000.):
        xx=x-east*distance/dx; yy=y-north*distance/dy
        inside=(xx>=0)&(xx<=z.shape[1]-1)&(yy>=0)&(yy<=z.shape[0]-1)&(speed>=.5)
        grid=torch.stack((2*xx/(z.shape[1]-1)-1,2*yy/(z.shape[0]-1)-1),dim=-1)
        sample=F.grid_sample(z[None,None].expand(len(u),-1,-1,-1),grid,align_corners=True)[:,0]
        relief=torch.where(inside,(sample-z)/distance,0.)
        fields.extend([relief*10,inside.float()]); upstream.append(relief); valid.append(inside)
    stacked=torch.stack(upstream)
    fields.extend([stacked.clamp_min(0).amax(0)*10,(-stacked).clamp_min(0).amax(0)*10])
    return torch.stack(static_fields),torch.stack(fields,dim=1)
