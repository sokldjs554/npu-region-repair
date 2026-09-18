"""Official CIFAR-10 binary input, checksummed and split without exact-image leakage."""
from pathlib import Path, PurePosixPath
import hashlib
import tarfile
import json
import tempfile
from .cifar_download import fetch_archive
import numpy as np
from nrr.artifacts import digest_file, digest_array, write_json
from nrr.data import Dataset

ARCHIVE_SHA256='c4a38c50a1bc5f3a1c5537f2155ab9d68f9f25eb1ed8d9ddda3db29a59bca1dd'
ARCHIVE_MD5='c32a1d4ab5d03f1284b67883e8d87530'
ARCHIVE_BYTES=170052171
URLS=('https://storage.googleapis.com/mirror.tensorflow.org/www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz',
      'https://mirror.tensorflow.org/www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz',
      'https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz')
BATCHES=tuple(f'cifar-10-batches-bin/data_batch_{i}.bin' for i in range(1,6))+('cifar-10-batches-bin/test_batch.bin',)

def ensure_archive(cache: str | Path,download: bool=True)->Path:
    cache=Path(cache)
    if cache.is_symlink():raise ValueError('Cache must not be a symlink')
    cache.mkdir(parents=True,exist_ok=True); path=cache/'cifar-10-binary.tar.gz'
    if path.is_symlink():raise ValueError('Archive must not be a symlink')
    if path.exists():
        if not path.is_file() or digest_file(path)!=ARCHIVE_SHA256:raise ValueError('Cached CIFAR checksum differs; preserve and inspect the file')
        return path
    if not download:raise FileNotFoundError('Verified CIFAR-10 archive is not cached')
    return fetch_archive(cache,path.name,ARCHIVE_SHA256,ARCHIVE_BYTES,URLS)

def read_archive(path, *, expected_sha256=ARCHIVE_SHA256,rows_per_file=10000):
    path=Path(path)
    if digest_file(path)!=expected_sha256:raise ValueError('CIFAR archive checksum differs')
    if expected_sha256==ARCHIVE_SHA256:
        h=hashlib.md5(usedforsecurity=False)
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1<<20),b''):h.update(block)
        if h.hexdigest()!=ARCHIVE_MD5:raise ValueError('CIFAR MD5 checksum differs')
    with tarfile.open(path,'r:gz') as archive:
        members={}
        for m in archive.getmembers():
            p=PurePosixPath(m.name)
            if p.is_absolute() or '..' in p.parts or m.issym() or m.islnk() or not (m.isfile() or m.isdir()):
                raise ValueError(f'Unsafe archive member: {m.name}')
            if m.name in members:raise ValueError('Duplicate archive member')
            members[m.name]=m
        xs=[];ys=[]
        for name in BATCHES:
            m=members.get(name)
            if m is None or not m.isfile() or m.size!=rows_per_file*3073:raise ValueError(f'Invalid binary member size: {name}')
            with archive.extractfile(m) as f:raw=f.read(m.size+1)
            if len(raw)!=m.size:raise ValueError('Truncated binary member size')
            records=np.frombuffer(raw,np.uint8).reshape(rows_per_file,3073)
            labels=records[:,0].astype(np.int64)
            if np.any(labels>9):raise ValueError('Invalid CIFAR class label')
            xs.append(records[:,1:].reshape(-1,3,32,32).copy());ys.append(labels)
    return np.concatenate(xs[:-1]),np.concatenate(ys[:-1]),xs[-1],ys[-1]

def make_split(x,y,tx,ty,*,train_per_class,val_per_class,seed,test_limit=None):
    for arr,labels in ((x,y),(tx,ty)):
        if arr.dtype!=np.uint8 or arr.shape[1:]!=(3,32,32) or arr.ndim!=4 or labels.shape!=(len(arr),):
            raise ValueError('Expected NCHW uint8 CIFAR images with labels')
        if not np.array_equal(np.unique(labels),np.arange(10)):raise ValueError('Expected ten classes')
    if min(train_per_class,val_per_class)<1:raise ValueError('Positive per-class sample sizes required')
    hashes=lambda a:[hashlib.sha256(image.tobytes()).hexdigest() for image in a]
    th=hashes(tx);xh=hashes(x);test_hashes=set(th)
    removed=[i for i,h in enumerate(xh) if h in test_hashes]
    byhash={}
    for i,h in enumerate(xh):byhash.setdefault(h,[]).append(i)
    conflicts={h for h,ids in byhash.items() if len(set(y[ids].tolist()))>1}
    rng=np.random.default_rng(seed);train=[];val=[];seen=set(test_hashes)|conflicts
    duplicate_skipped=0
    for c in range(10):
        chosen=[]
        for i in rng.permutation(np.flatnonzero(y==c)):
            h=xh[i]
            if h in seen:duplicate_skipped+=1;continue
            seen.add(h);chosen.append(int(i))
            if len(chosen)==train_per_class+val_per_class:break
        if len(chosen)<train_per_class+val_per_class:raise ValueError(f'Insufficient clean data for class {c}')
        val.extend(chosen[:val_per_class]);train.extend(chosen[val_per_class:])
    train=np.sort(train);val=np.sort(val)
    ti=np.arange(len(tx))
    if test_limit is not None:
        if not isinstance(test_limit,int) or test_limit<10 or test_limit>len(tx) or test_limit%10:
            raise ValueError('Diagnostic test_limit must be a multiple of ten within the official test size')
        ti=np.sort(np.concatenate([np.random.default_rng(seed+c).permutation(np.flatnonzero(ty==c))[:test_limit//10] for c in range(10)]))
    xx=np.concatenate((x[train],x[val],tx[ti])).astype(np.float32)/np.float32(255)
    yy=np.concatenate((y[train],y[val],ty[ti])).astype(np.int64)
    a=len(train);b=a+len(val)
    data=Dataset(xx,yy,np.arange(a),np.arange(a,b),np.arange(b,len(xx)),
                 'CIFAR-10 official binary; fixed training subset',
                 'fixed_training_subset_full_official_test' if test_limit is None else 'fixed_training_and_test_subsets')
    meta={'schema':'nrr.cifar-subset.v1','original_indices':{'train':train.tolist(),'val':val.tolist(),'test':ti.tolist()},
          'source_train_images':len(x),'source_test_images':len(tx),'split_seed':seed,
          'removed_train_matching_test':len(removed),'removed_train_matching_test_indices':removed,
          'conflicting_training_hash_groups':len(conflicts),'duplicate_skip_visits':duplicate_skipped,
          'exact_duplicate_policy':'keep official test; remove matching train; use one representative per training image hash',
          'near_duplicates_checked':False,'normalization':'uint8 / 255, no augmentation',
          'train_per_class':train_per_class,'val_per_class':val_per_class,'test_limit':test_limit,
          'data_sha256':digest_array(data.x,data.y),'scope':data.scope,'source':data.source}
    return data,meta

def prepare(cache,out,protocol,download=True):
    out=Path(out)
    if out.exists() or out.is_symlink():raise FileExistsError('Prepared dataset output already exists')
    def progress(event,**kw):
        print('[cifar] '+json.dumps({'event':event,**kw},ensure_ascii=False,allow_nan=False),flush=True)
    progress('prepare_start',protocol_id=protocol.protocol_id)
    archive=ensure_archive(cache,download);progress('archive_read',archive_bytes=archive.stat().st_size)
    raw=read_archive(archive);progress('split_start',source_train_images=len(raw[0]),source_test_images=len(raw[2]))
    data,meta=make_split(*raw,train_per_class=protocol.train_per_class,val_per_class=protocol.val_per_class,
                         seed=protocol.split_seed,test_limit=protocol.test_limit)
    out.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.'+out.name+'-',dir=out.parent))
    progress('npz_write',train_images=len(data.train),validation_images=len(data.val),test_images=len(data.test))
    try:
        np.savez(staging/'data.npz',x=data.x,y=data.y,train=data.train,val=data.val,test=data.test)
        meta.update(archive_sha256=digest_file(archive),archive_md5=ARCHIVE_MD5,
                    npz_sha256=digest_file(staging/'data.npz'),protocol_id=protocol.protocol_id)
        write_json(staging/'dataset.json',meta)
        if out.exists() or out.is_symlink():raise FileExistsError('Prepared output appeared during write')
        staging.rename(out)
    except BaseException as exc:
        progress('prepare_failed',exception=type(exc).__name__,message=str(exc));raise
    progress('prepare_complete',protocol_id=protocol.protocol_id,npz_sha256=meta['npz_sha256'],
             train_images=len(data.train),validation_images=len(data.val),test_images=len(data.test))
    return out/'data.npz'
