"""Dedicated wind decoder: shared trunk with Cartesian, polar and hybrid heads."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self,width,dilation=1):
        super().__init__()
        self.ops=nn.Sequential(nn.Conv2d(width,width,3,padding=dilation,dilation=dilation),
            nn.GroupNorm(4,width),nn.SiLU(),nn.Conv2d(width,width,3,padding=1))

    def forward(self,x):
        return x+self.ops(x)*.5


class WindNet(nn.Module):
    def __init__(self,nin,width,head,arch):
        super().__init__()
        self.head=head; self.arch=arch
        self.stem=nn.Conv2d(nin,width,3,padding=1)
        self.body=nn.Sequential(Block(width),Block(width,2))
        if arch=='pyramid':
            self.context=nn.Sequential(Block(width),Block(width,2))
        if arch=='spectral':
            self.context=nn.Conv2d(width,width,1)
        self.out=nn.Conv2d(width,3,1)
        nn.init.normal_(self.out.weight,std=.005)
        with torch.no_grad(): self.out.bias.copy_(torch.tensor([.5,.1,.2]))
        if head=='hybrid': self.direct=nn.Conv2d(width,2,1)

    def forward(self,x):
        h=self.body(self.stem(x))
        if self.arch=='pyramid':
            h=h+F.interpolate(self.context(F.avg_pool2d(h,3,stride=3)),size=h.shape[-2:],mode='bilinear',align_corners=False)
        if self.arch=='spectral':
            # Reflect to avoid a periodic seam; truncate to low Fourier frequencies.
            p=torch.cat([h,h.flip(-1)],-1)
            p=torch.cat([p,p.flip(-2)],-2)
            f=torch.fft.rfft2(p); kh,kw=8,6
            keep=torch.zeros_like(f.real); keep[...,:kh,:kw]=1; keep[...,-kh+1:,:kw]=1
            h=h+self.context(torch.fft.irfft2(f*keep,s=p.shape[-2:])[...,:h.shape[-2],:h.shape[-1]].contiguous())
        a=self.out(h)
        if self.head=='uv': return a[:,:2],None
        if self.head=='angle':
            theta=math.pi*a[:,1:2]; direction=torch.cat([theta.cos(),theta.sin()],dim=1)
        else:
            direction=F.normalize(a[:,1:3],dim=1,eps=1e-4)
        speed=F.softplus(a[:,:1]); wind=speed*direction
        return ((wind+self.direct(h))*.5 if self.head=='hybrid' else wind),(speed,direction)


def loss(out,target,channel_weights,aux=0.,extreme=0.):
    pred,polar=out
    speed=target.norm(dim=1,keepdim=True)
    emphasis=1+extreme*(speed.detach()>2.).float()
    err=((pred-target).square()*emphasis).mean((0,2,3))
    objective=(err*channel_weights).sum()
    if aux and polar is not None:
        magnitude,direction=polar
        weight=(speed/.25).clamp(0,1)
        cosine=(direction*target/speed.clamp_min(1e-4)).sum(dim=1,keepdim=True)
        objective=objective+aux*((magnitude-speed).square().mean()+(weight*(1-cosine)).mean())
    return objective
