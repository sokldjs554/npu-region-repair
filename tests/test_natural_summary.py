import copy
import numpy as np
import pytest
from nrr_natural.aggregate import aggregate,paired_difference
from nrr.experiment import METHODS

def record(seed,method,budget,pred):
    y=[0,1,0,1];score={'n':4,'correct':sum(a==b for a,b in zip(pred,y)),
                      'accuracy':sum(a==b for a,b in zip(pred,y))/4,'predictions':pred,'subset_sha256':'same','ids':[1,2,3,4]}
    return {'seed':seed,'method':method,'budget_steps':budget,'fp32':score,'tflite_cpu':None,
            'test_subset_sha256':'same','test_ids':[1,2,3,4],'parameters':12,'compiler':None}

def result_fixture():
    records=[]
    for s in (17,29,43):
        records.append(record(s,'teacher',None,[0,1,0,1]))
        for m in METHODS:records.append(record(s,m,80,[0,1,0,0] if m=='regionwise' else [0,1,0,1]))
    return {'records':records,'protocol':{'seeds':[17,29,43],'budget_steps':[80],'methods':list(METHODS)},
            'test_subset_sha256':'same','test_ids':[1,2,3,4],'test_labels':[0,1,0,1],
            'backend':'cpu-check','status':'mechanism_check_complete'}

def test_paired_difference_and_no_pooled_seeds():
    r=paired_difference([0,1,0,0],[0,1,0,1],[0,1,0,1])
    assert r['delta_pp']==-25 and r['n']==4 and r['different_predictions']==1
    a=aggregate(result_fixture())
    assert a['training_replicates']==3 and a['unique_test_images']==4
    rr=next(x for x in a['paired_seed_summaries'] if x['candidate']=='regionwise' and x['reference']=='small_student')
    assert rr['mean_delta_pp']==-25 and rr['seed_sd_pp']==0 and rr['n_seeds']==3

@pytest.mark.parametrize('mutation',['duplicate','wrong_n','wrong_correct','wrong_subset','unknown_method'])
def test_corrupted_rows_are_rejected(mutation):
    r=result_fixture()
    if mutation=='duplicate':r['records'].append(copy.deepcopy(r['records'][0]))
    elif mutation=='wrong_n':r['records'][0]['fp32']['n']=40
    elif mutation=='wrong_correct':r['records'][0]['fp32']['correct']=0
    elif mutation=='wrong_subset':r['records'][0]['test_subset_sha256']='other'
    else:r['records'][0]['method']='unregistered'
    with pytest.raises(ValueError):aggregate(r)

def test_partial_run_not_called_replicated_result():
    r=result_fixture();r['records']=r['records'][:-1];r['status']='partial'
    a=aggregate(r)
    assert a['complete'] is False and a['paired_seed_summaries']==[]
