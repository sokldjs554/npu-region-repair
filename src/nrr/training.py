"""Real CPU training with budget accounting and validation-only selection."""
from __future__ import annotations
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from .artifacts import state_digest, digest_array
from .budget import Budget,batch_charge,positive_int
from .data import Dataset
from .models import RegionCNN,forward_macs


def batch_stream(train_ids: np.ndarray,seed: int,steps: int,batch_size: int)->np.ndarray:
    positive_int(steps,'steps'); positive_int(batch_size,'batch_size')
    if not len(train_ids): raise ValueError('Empty train IDs')
    rng=np.random.default_rng(seed)
    n=steps*batch_size
    return np.concatenate([rng.permutation(train_ids) for _ in range(math.ceil(n/len(train_ids)))])[:n].reshape(steps,batch_size)


@torch.no_grad()
def evaluate(model: torch.nn.Module,data: Dataset,ids: np.ndarray)->dict:
    model.eval(); predictions=[]
    for chunk in np.array_split(ids,max(1,math.ceil(len(ids)/128))):
        out=model(torch.from_numpy(data.x[chunk]))
        if out.shape!=(len(chunk),data.classes) or not torch.isfinite(out).all():
            raise ValueError('Invalid classification output')
        predictions.extend(out.argmax(1).tolist())
    y=data.y[ids]; correct=int(np.equal(y,predictions).sum())
    confusion=np.zeros((data.classes,data.classes),dtype=np.int64)
    np.add.at(confusion,(y,np.asarray(predictions)),1)
    return {'n':len(ids),'correct':correct,'accuracy':correct/len(ids),'predictions':predictions,
            'labels':y.tolist(),'ids':ids.tolist(),'subset_sha256':digest_array(ids,data.x[ids],y),
            'confusion_matrix':confusion.tolist()}


def _state(model): return {k:v.detach().clone() for k,v in model.state_dict().items()}


def train_teacher(model: RegionCNN,data: Dataset,steps: int,seed: int,batch_size: int=32,lr: float=.04)->dict:
    positive_int(steps,'steps')
    stream=batch_stream(data.train,seed,steps,batch_size)
    optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=.9,weight_decay=1e-4)
    start=time.perf_counter(); best=-1; chosen=0; state=_state(model); log=[]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for step,ids in enumerate(stream,1):
            model.train(); rate=lr*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/steps)))
            for g in optimizer.param_groups:g['lr']=rate
            optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(torch.from_numpy(data.x[ids])),torch.from_numpy(data.y[ids]))
            if not torch.isfinite(loss):raise ValueError('Nonfinite teacher loss')
            loss.backward();optimizer.step()
            if step%max(1,steps//10)==0 or step==steps:
                val=evaluate(model,data,data.val)
                log.append({'step':step,'train_loss':float(loss.detach()),'validation_accuracy':val['accuracy']})
                if val['accuracy']>best: best=val['accuracy'];state=_state(model);chosen=step
    model.load_state_dict(state);model.eval()
    return {'steps':steps,'images':steps*batch_size,'selected_step':chosen,'selection_split':'validation',
            'validation_accuracy':best,'seconds':time.perf_counter()-start,'log':log,
            'batch_stream_sha256':digest_array(stream),'state_sha256':state_digest(model),
            'forward_macs_per_image':forward_macs(model,tuple(data.x.shape[1:]))}


def fit_budgeted(model: RegionCNN,teacher: RegionCNN,data: Dataset,stream: np.ndarray,
                 cap: int,seed: int,lr: float=.04)->dict:
    if stream.ndim!=2 or not set(stream.flat).issubset(set(data.train)):
        raise ValueError('Training stream must contain train IDs only')
    if not math.isfinite(lr) or lr<=0: raise ValueError('lr must be finite and positive')
    sm=forward_macs(model,tuple(data.x.shape[1:]));tm=forward_macs(teacher,tuple(data.x.shape[1:]))
    cost=batch_charge(sm,tm,stream.shape[1]);budget=Budget(int(cap))
    if not budget.can_charge(cost):raise ValueError('Budget cannot pay for one batch')
    if len(stream)<cap//cost:raise ValueError('Batch stream shorter than budget permits')
    teacher.eval();teacher_hash=state_digest(teacher)
    initial_hash=state_digest(model);optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=.9,weight_decay=1e-4)
    start=time.perf_counter();initial=evaluate(model,data,data.val)
    best=initial['accuracy'];selected_step=0;selected=_state(model)
    checks=[{'step':0,'budget_spent':0,'accuracy':best,'selection_split':'validation'}]
    threshold=1;losses=[];steps=0
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for ids in stream:
            if not budget.can_charge(cost):break
            model.train();budget.charge(cost);steps+=1
            rate=lr*(.1+.9*.5*(1+math.cos(math.pi*budget.spent/cap)))
            for group in optimizer.param_groups:group['lr']=rate
            x=torch.from_numpy(data.x[ids]);y=torch.from_numpy(data.y[ids])
            with torch.no_grad():
                tf=teacher.features(x);tz=teacher.head(torch.flatten(teacher.pool(tf),1))
            sf=model.features(x);sz=model.head(torch.flatten(model.pool(sf),1))
            ce=F.cross_entropy(sz,y)
            kd=F.kl_div(F.log_softmax(sz/2,dim=1),F.softmax(tz/2,dim=1),reduction='batchmean')*4
            feature=F.mse_loss(sf,tf)
            loss=.5*ce+.5*kd+.1*feature
            if not torch.isfinite(loss):raise ValueError('Nonfinite recovery loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            losses.append({'step':steps,'spent':budget.spent,'loss':float(loss.detach()),'lr':rate})
            if budget.spent>=cap*threshold/5 or not budget.can_charge(cost):
                val=evaluate(model,data,data.val)
                checks.append({'step':steps,'budget_spent':budget.spent,'accuracy':val['accuracy'],'selection_split':'validation'})
                if val['accuracy']>best:best=val['accuracy'];selected_step=steps;selected=_state(model)
                threshold+=1
        last_hash=state_digest(model)
        model.load_state_dict(selected);model.eval()
    if state_digest(teacher)!=teacher_hash:raise RuntimeError('Teacher was mutated')
    return {'updates':steps,'images':steps*stream.shape[1],'teacher_images':steps*stream.shape[1],
            'budget_cap':cap,'budget_spent':budget.spent,'unused_budget':budget.remaining,
            'budget_unit':'conv_linear_MAC_proxy: (3*student+teacher)*images',
            'exclusions':['normalization','elementwise ops','memory','validation','quantization','actual backward kernels'],
            'student_forward_macs':sm,'teacher_forward_macs':tm,'selected_step':selected_step,
            'validation_accuracy':best,'checkpoints':checks,'training':losses,
            'test_accesses_during_training':0,'seconds':time.perf_counter()-start,
            'initial_state_sha256':initial_hash,'last_state_sha256':last_hash,
            'selected_state_sha256':state_digest(model),'stream_prefix_sha256':digest_array(stream[:steps])}
