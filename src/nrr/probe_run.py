"""Run external conversion/compilation on six *already trained* fixed models."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import torch
from .artifacts import digest_file, environment, reserve_output, write_json
from .compiler import CompilerUnavailable, check_external_dependencies, require_external_baseline, run_vela
from .saved_probe import load_saved_inputs, preflight
from .tflite_bridge import convert_tflite, evaluate_tflite
from .training import evaluate
from .probe_report import render_probe


def run_saved_probe(saved_run: str | Path, out: str | Path, data_path: str | Path | None = None,
                    executable: str = 'vela', accelerator: str = 'ethos-u55-256') -> dict:
    source=Path(saved_run).resolve();target=Path(out).resolve()
    if source==target or source in target.parents:
        raise ValueError('Probe output must not be inside the immutable saved run')
    bound=load_saved_inputs(source,data_path)
    dest=reserve_output(target)
    report={'schema':'nrr.saved-compiler-probe.v1','created_at':datetime.now(timezone.utc).isoformat(),
            'status':'running','exit_code':2,'records':{},'new_training_updates':0,
            'compiler_executed':False,'npu_hardware_executed':False,'npu_latency_ms':None,
            'research_claim_validated':False,'source_result_sha256':bound.source_snapshot['result.json'],
            'source_manifest_sha256':bound.source_snapshot['manifest.json'],
            'data_sha256':bound.result['data']['data_sha256'],
            'test_ids':bound.data.test.tolist(),'calibration_ids':bound.calibration_ids.tolist(),
            'validation_parity_ids':bound.data.val[:16].tolist(),
            'scope':'saved small-CNN probe; compiler topology plus host CPU accuracy, no NPU execution',
            'environment':environment(Path(__file__).resolve().parents[2]),'source_preserved':False}
    old_threads=torch.get_num_threads();torch.set_num_threads(1)
    def save():
        write_json(dest/'probe.json',report)
        render_probe(report,dest)
    try:
        report['preflight']=preflight(source,dest/'inputs',data_path)
        save()
        try:
            check_external_dependencies(executable)
        except CompilerUnavailable as exc:
            report.update(status='blocked_environment',exit_code=3,reason=str(exc))
            return report
        failed=[]
        for name in bound.checkpoints:
            directory=dest/name;directory.mkdir()
            row={'status':'running','checkpoint_sha256':digest_file(bound.checkpoints[name]),
                 'compiler_attempted':False}
            report['records'][name]=row;save()
            try:
                model=bound.load_model(name)
                row['config']=model.config
                row['fp32_cpu']=evaluate(model,bound.data,bound.data.test)
                original=(bound.result['baseline'] if name=='teacher' else bound.result['methods'][name])['fp32_test']
                row['original_fp32_cpu_correct']=original['correct']
                row['fp32_prediction_changes_from_original']=sum(a!=b for a,b in zip(
                    original['predictions'],row['fp32_cpu']['predictions'],strict=True))
                row['export']=convert_tflite(model,bound.data.x[bound.calibration_ids],directory/'tflite',
                                            parity_images=bound.data.x[bound.data.val[:16]])
                row['tflite_cpu']=evaluate_tflite(row['export']['path'],bound.data.x[bound.data.test],bound.data.y[bound.data.test])
                row['compiler_attempted']=True
                row['compiler']=run_vela(row['export']['path'],directory/'vela',executable=executable,accelerator=accelerator)
                row['status']='observed';report['compiler_executed']=True
                if name=='teacher':
                    try:
                        require_external_baseline(row['compiler'])
                        report['baseline_gate']={'passed':True,'reason':'Mixed compiler graph observed; host tensor paths include exact r0 and r1 tokens. This is name-based attribution, not a universal graph proof.'}
                    except ValueError as exc:
                        report['baseline_gate']={'passed':False,'reason':str(exc)}
            except Exception as exc:
                row.update(status='failed',exception=type(exc).__name__,reason=str(exc))
                write_json(directory/'failed.json',row)
                failed.append(name)
            save()
        baseline=report['records'].get('teacher',{}).get('compiler')
        if baseline:
            for name,row in report['records'].items():
                if name!='teacher' and row.get('status')=='observed':
                    row['compiler_delta']={k:row['compiler'][k]-baseline[k] for k in
                                         ('host_operator_count','npu_partition_count','boundary_tensor_count')}
        if failed:
            report.update(status='partial',exit_code=2,reason='Failed fixed arms: '+', '.join(failed))
        elif not report.get('baseline_gate',{}).get('passed'):
            report.update(status='inconclusive',exit_code=4,
                          reason='Models compiled, but the original mixed regions could not be attributed; no NPU repair claim.')
        else:
            report.update(status='observed',exit_code=0,
                          reason='All fixed arms compiled and the baseline gate passed. No winning method or NPU performance claim.')
        return report
    except Exception as exc:
        report.update(status='failed',exit_code=2,reason=str(exc),exception=type(exc).__name__)
        return report
    finally:
        try:
            bound.assert_preserved();report['source_preserved']=True
        except Exception as exc:
            report.update(status='failed',exit_code=2,reason=f'Source integrity changed: {exc}',source_preserved=False)
        report['environment_after']=environment(Path(__file__).resolve().parents[2])
        save()
        (dest/'.running').unlink(missing_ok=True)
        paths={str(p.relative_to(dest)):{'sha256':digest_file(p),'bytes':p.stat().st_size}
               for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='probe-manifest.json'}
        write_json(dest/'probe-manifest.json',{'files':paths,'scope':'file integrity only; inspect probe.json status'})
        torch.set_num_threads(old_threads)
