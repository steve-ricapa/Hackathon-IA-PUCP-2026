"""Matched temporal and localized starter adaptations."""
import numpy as np
import pandas as pd
import torch
from torch import nn


def weights(times, cutoff, half_life):
    age=np.asarray((cutoff-times).total_seconds())/86400
    if not len(age) or np.any(age<=0) or half_life<=0:
        raise ValueError('Expected strictly pre-cut times and positive half-life')
    w=np.exp2(-age/half_life); w/=w.mean()
    return w.astype('float32'),dict(min=float(w.min()),max=float(w.max()),
                                    effective_hours=float(w.sum()**2/(w@w)))


def batches(n, seed, *, times=None, cutoff=None, days=None):
    order=np.arange(n)
    if days is not None:
        order=order[times>=cutoff-pd.Timedelta(days=days)]
    if not len(order): raise ValueError('No available training samples')
    rng=np.random.RandomState(seed)
    while True:
        rng.shuffle(order)
        for i in range(0,len(order),32): yield order[i:i+32].copy()


class Readout(nn.Module):
    def __init__(self, starter, local, width=4):
        super().__init__()
        self.base=starter.requires_grad_(False)
        last=self.base.dec[-1]; self.compress=nn.Conv2d(last.in_channels,width,1,bias=False)
        self.shared=nn.Conv2d(width,last.out_channels,1,bias=False)
        nn.init.zeros_(self.shared.weight)
        self.delta=nn.Parameter(torch.zeros(last.out_channels,width,*starter.dem.shape[-2:])) if local else None

    def forward(self,x,h,m):
        # Capture exactly the input to the final readout, without cloning starter logic.
        saved=[]
        handle=self.base.dec[-1].register_forward_pre_hook(lambda _,args: saved.append(args[0]))
        try:
            with torch.no_grad(): out=self.base(x,h,m)
        finally:
            handle.remove()
        z=self.compress(saved[0]); correction=self.shared(z)
        if self.delta is not None:
            correction=correction+torch.einsum('nkhw,okhw->nohw',z,self.delta)
        return out+correction.permute(0,2,3,1)

    def penalty(self):
        return self.delta.square().mean() if self.delta is not None else 0.
