"""Fixed split IDs; never use test data for calibration or selection."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from .artifacts import digest_array, digest_file


@dataclass
class Dataset:
    x: np.ndarray
    y: np.ndarray
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    source: str
    scope: str='user_supplied_data_not_independently_verified'

    def __post_init__(self):
        if self.x.dtype!=np.float32 or self.x.ndim!=4 or not np.isfinite(self.x).all():
            raise ValueError('x must be finite NCHW float32')
        if self.x.shape[2]!=self.x.shape[3] or self.x.shape[2]<8 or self.x.shape[2]%4:
            raise ValueError('Square image side must be >=8 and divisible by 4')
        if self.x.shape[1] not in (1,3): raise ValueError('Expected one or three channels')
        if self.y.ndim!=1 or len(self.y)!=len(self.x) or not np.issubdtype(self.y.dtype,np.integer):
            raise ValueError('y must contain one integer class per image')
        labels=np.unique(self.y)
        if len(labels)<2 or not np.array_equal(labels,np.arange(len(labels))):
            raise ValueError('Class labels must be contiguous starting at zero')
        self.y=self.y.astype(np.int64,copy=False)
        seen=set()
        for name in ('train','val','test'):
            ids=getattr(self,name)
            if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or not len(ids):
                raise ValueError(f'{name} must contain nonempty integer IDs')
            curr=set(ids.tolist())
            if len(curr)!=len(ids) or min(curr)<0 or max(curr)>=len(self.x):
                raise ValueError(f'Invalid or duplicate {name} IDs')
            if curr&seen: raise ValueError('Split IDs overlap')
            seen|=curr
        if len(seen)!=len(self.x): raise ValueError('Splits must cover all images exactly once')
        hashes={}
        for name in ('train','val','test'):
            for ix in getattr(self,name):
                key=digest_array(self.x[ix])
                prior=hashes.setdefault(key,name)
                if prior!=name: raise ValueError('Duplicate image content across splits')

    @property
    def classes(self)->int: return int(self.y.max())+1

    def manifest(self)->dict:
        return {'source':self.source,'scope':self.scope,'shape':list(self.x.shape),'classes':self.classes,
                'data_sha256':digest_array(self.x,self.y),
                **{name:{'n':len(getattr(self,name)),'ids':getattr(self,name).tolist(),
                          'sha256':digest_array(getattr(self,name))} for name in ('train','val','test')}}


def load_digits_split(seed: int=17)->Dataset:
    d=load_digits()
    x=(d.images[:,None]/16).astype(np.float32); y=d.target.astype(np.int64)
    train,remaining=train_test_split(np.arange(len(y)),test_size=.4,stratify=y,random_state=seed)
    val,test=train_test_split(remaining,test_size=.5,stratify=y[remaining],random_state=seed+1)
    return Dataset(x,y,np.sort(train),np.sort(val),np.sort(test),
                   'sklearn.datasets.load_digits (UCI optical digits copy)', 'offline_mechanism_check_not_NPU_benchmark')


def load_npz(path: str | Path)->Dataset:
    p=Path(path)
    with np.load(p,allow_pickle=False) as z:
        keys={'x','y','train','val','test'}
        if not keys.issubset(z.files): raise ValueError(f'Missing keys: {sorted(keys-set(z.files))}')
        arrays={k:z[k].copy() for k in keys}
    return Dataset(**arrays,source=str(p),scope='user_npz_'+digest_file(p))
