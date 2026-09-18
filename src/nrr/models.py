"""Fixed diagnostic architecture. Candidate regions are NOT assumed unsupported."""
from __future__ import annotations
from collections import OrderedDict
import copy
import torch
from torch import nn


def original_region(width: int)->nn.Sequential:
    return nn.Sequential(OrderedDict(conv=nn.Conv2d(width,width,3,padding=1,bias=False),
                        norm=nn.GroupNorm(1,width),act=nn.GELU()))


def bottleneck_region(width: int)->nn.Sequential:
    hidden=max(2,width//2)
    return nn.Sequential(OrderedDict(reduce=nn.Conv2d(width,hidden,1,bias=False),
                        bn0=nn.BatchNorm2d(hidden),relu0=nn.ReLU(),
                        conv=nn.Conv2d(hidden,width,3,padding=1,bias=False),
                        bn1=nn.BatchNorm2d(width),relu1=nn.ReLU()))


class RegionCNN(nn.Module):
    def __init__(self,channels: int=1,classes: int=10,width: int=16):
        super().__init__()
        if width<4 or width%2: raise ValueError('width must be even and >=4')
        self.config={'channels':channels,'classes':classes,'width':width,'kind':'original'}
        self.stem=nn.Sequential(nn.Conv2d(channels,width,3,padding=1),nn.ReLU())
        self.r0=original_region(width); self.r1=original_region(width)
        self.pool=nn.AdaptiveAvgPool2d((4,4)); self.head=nn.Linear(width*16,classes)

    def features(self,x): return self.r1(self.r0(self.stem(x)))
    def forward(self,x): return self.head(torch.flatten(self.pool(self.features(x)),1))


def make_candidate(teacher: RegionCNN,kind: str,seed: int)->RegionCNN:
    if kind not in ('continuation','operatorwise','regionwise','single_region','small_student'):
        raise ValueError(f'Unknown candidate: {kind}')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        cfg=teacher.config; w=cfg['width']
        model=(RegionCNN(cfg['channels'],cfg['classes'],w) if kind=='small_student' else copy.deepcopy(teacher))
        if kind=='operatorwise':
            for name in ('r0','r1'):
                block=getattr(model,name)
                block.norm=nn.BatchNorm2d(w)
                block.act=nn.ReLU()
        elif kind in ('regionwise','small_student','single_region'):
            for name in (('r0',) if kind=='single_region' else ('r0','r1')):
                setattr(model,name,bottleneck_region(w))
        model.config={**cfg,'kind':kind}
    return model.eval()


def model_from_config(config: dict)->RegionCNN:
    original=RegionCNN(config['channels'],config['classes'],config['width'])
    return original if config['kind']=='original' else make_candidate(original,config['kind'],0)


def forward_macs(model: nn.Module,input_shape: tuple[int,...])->int:
    """Conv/linear multiply-accumulates only; no claim of actual cycle count."""
    total=0
    def hook(m,inputs,output):
        nonlocal total
        if isinstance(m,nn.Conv2d):
            total+=output.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
        elif isinstance(m,nn.Linear): total+=output.numel()*m.in_features
    handles=[m.register_forward_hook(hook) for m in model.modules() if isinstance(m,(nn.Conv2d,nn.Linear))]
    modes={m:m.training for m in model.modules()}
    try:
        model.eval()
        with torch.no_grad(): model(torch.zeros((1,*input_shape)))
    finally:
        for h in handles: h.remove()
        for m,mode in modes.items(): m.training=mode
    return int(total)
