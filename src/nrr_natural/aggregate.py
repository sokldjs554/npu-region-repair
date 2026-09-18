"""Paired comparisons within a seed, descriptive variation across training seeds."""
import math
import numpy as np

def paired_difference(candidate,reference,labels):
    a=np.asarray(candidate);b=np.asarray(reference);y=np.asarray(labels)
    if a.ndim!=1 or a.shape!=b.shape or a.shape!=y.shape or not len(y):raise ValueError('Paired arrays must align')
    diff=((a==y).astype(np.float64)-(b==y).astype(np.float64))*100
    se=float(diff.std(ddof=1)/np.sqrt(len(y))) if len(y)>1 else None
    delta=float(diff.mean())
    return {'n':len(y),'delta_pp':delta,'paired_se_pp':se,'descriptive_wald95_pp':[delta-1.96*se,delta+1.96*se] if se is not None else None,'different_predictions':int(np.sum(a!=b)),'candidate_only_correct':int(np.sum((a==y)&(b!=y))),'reference_only_correct':int(np.sum((a!=y)&(b==y))),'interpretation':'within-seed image-paired descriptive interval; not an equivalence test or multiplicity-corrected claim'}

def _validate_score(score,labels,subset,ids):
    n=len(labels)
    if not isinstance(score,dict) or type(score.get('n')) is not int or score['n']!=n:raise ValueError('Incorrect evaluation n')
    pred=score.get('predictions')
    if not isinstance(pred,list) or len(pred)!=n or any(type(x) is not int or not 0<=x<=max(labels) for x in pred):raise ValueError('Invalid prediction vector')
    actual=sum(a==b for a,b in zip(pred,labels,strict=True))
    if type(score.get('correct')) is not int or actual!=score['correct']:raise ValueError('Incorrect evaluation correct count')
    if not isinstance(score.get('accuracy'),(int,float)) or not math.isfinite(score['accuracy']) or abs(actual/n-score['accuracy'])>1e-12:raise ValueError('Incorrect evaluation accuracy')
    if score.get('subset_sha256')!=subset or score.get('ids')!=ids:raise ValueError('Evaluation subset differs')

def aggregate(result):
    protocol=result['protocol'];seeds=protocol['seeds'];budgets=protocol['budget_steps'];methods=protocol['methods']
    y=result['test_labels'];ids=result['test_ids'];subset=result['test_subset_sha256']
    if len(y)!=len(ids) or not len(y):raise ValueError('Missing test labels/IDs')
    expected={(s,None,'teacher') for s in seeds}|{(s,b,m) for s in seeds for b in budgets for m in methods}
    records={};metric='tflite_cpu' if result['backend']=='external' else 'fp32'
    for row in result['records']:
        key=(row['seed'],row['budget_steps'],row['method'])
        if key not in expected or key in records:raise ValueError('Duplicate or unregistered seed/budget/method')
        if row['test_subset_sha256']!=subset or row['test_ids']!=ids:raise ValueError('Test subset differs across rows')
        _validate_score(row['fp32'],y,subset,ids)
        if result['backend']=='external':
            _validate_score(row.get('tflite_cpu'),y,subset,ids)
            comp=row.get('compiler')
            if not comp or comp.get('status')!='observed' or comp.get('hardware_execution') is not False:raise ValueError('Missing observed compiler record')
            for field in ('host_operator_count','npu_partition_count','boundary_tensor_count'):
                if type(comp.get(field)) is not int or comp[field]<0:raise ValueError('Invalid compiler count')
        records[key]=row
    complete=set(records)==expected and result['status'] in ('observed','mechanism_check_complete')
    contrasts=[]
    for s in seeds:
        for b in budgets:
            for ref in ('operatorwise','small_student'):
                a=records.get((s,b,'regionwise'));rr=records.get((s,b,ref))
                if a is not None and rr is not None:
                    contrasts.append({'seed':s,'budget_steps':b,'candidate':'regionwise','reference':ref,'metric':metric,**paired_difference(a[metric]['predictions'],rr[metric]['predictions'],y)})
    paired=[];groups=[]
    if complete:
        for b in budgets:
            for ref in ('operatorwise','small_student'):
                values=[x['delta_pp'] for x in contrasts if x['budget_steps']==b and x['reference']==ref]
                paired.append({'budget_steps':b,'candidate':'regionwise','reference':ref,'metric':metric,'n_seeds':len(values),'per_seed_delta_pp':values,'mean_delta_pp':float(np.mean(values)),'seed_sd_pp':float(np.std(values,ddof=1)) if len(values)>1 else None})
            for m in methods:
                values=[records[(s,b,m)][metric]['accuracy']*100 for s in seeds]
                groups.append({'budget_steps':b,'method':m,'metric':metric,'n_seeds':len(seeds),'mean_accuracy_pct':float(np.mean(values)),'seed_sd_pct':float(np.std(values,ddof=1)) if len(values)>1 else None})
    return {'complete':complete,'expected_records':len(expected),'observed_records':len(records),'training_replicates':len(seeds),'unique_test_images':len(y),'metric':metric,'missing_keys':[list(k) for k in sorted(expected-set(records),key=str)],'paired_within_seed':contrasts,'paired_seed_summaries':paired,'method_seed_summaries':groups,'winner_selected':False,'multiplicity_correction_applied':False,'caution':'Same test images reused across seeds. Seed SD describes training variation; no pooled n, significance or equivalence claim.'}
