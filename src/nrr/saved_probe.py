"""Bind an external probe to saved weights and original data, without retraining."""
from __future__ import annotations
from dataclasses import dataclass
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from sklearn.datasets import load_digits
from .artifacts import digest_array, digest_file, environment, reserve_output, state_digest, write_json
from .data import Dataset, load_npz
from .experiment import METHODS, verify_artifacts
from .models import model_from_config

DIGITS_SOURCE = 'sklearn.datasets.load_digits (UCI optical digits copy)'


def read_json(path: Path) -> dict:
    def reject(value):
        raise ValueError(f'Nonfinite JSON constant: {value}')
    try:
        result = json.loads(path.read_text(encoding='utf-8'), parse_constant=reject)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'Cannot read {path.name}: {exc}') from exc
    if not isinstance(result, dict):
        raise ValueError(f'{path.name} must be a JSON object')
    return result


@dataclass
class SavedInputs:
    root: Path
    result: dict
    data: Dataset
    calibration_ids: np.ndarray
    checkpoints: dict[str, Path]
    source_snapshot: dict[str, str]

    def current_source_snapshot(self) -> dict[str, str]:
        return {name: digest_file(self.root / name) for name in self.source_snapshot}

    def assert_preserved(self) -> None:
        if self.current_source_snapshot() != self.source_snapshot:
            raise RuntimeError('Saved source changed during probe')

    def load_model(self, name: str) -> torch.nn.Module:
        path = self.checkpoints[name]
        relative = str(path.relative_to(self.root))
        if digest_file(path) != self.source_snapshot[relative]:
            raise ValueError('Checkpoint integrity changed before loading')
        blob = torch.load(path, weights_only=True, map_location='cpu')
        if not isinstance(blob, dict) or set(blob) != {'config', 'state_dict'}:
            raise ValueError('Checkpoint requires config and state_dict only')
        cfg = blob['config']
        if not isinstance(cfg, dict) or set(cfg) != {'channels', 'classes', 'width', 'kind'}:
            raise ValueError('Unsupported checkpoint config')
        if any(type(cfg[k]) is not int for k in ('channels', 'classes', 'width')):
            raise ValueError('Model dimensions must be integers')
        if (cfg['channels'] != self.data.x.shape[1] or cfg['classes'] != self.data.classes
                or not 4 <= cfg['width'] <= 512 or cfg['width'] % 2):
            raise ValueError('Checkpoint dimensions disagree with dataset or supported bounds')
        expected_kind = 'original' if name == 'teacher' else name
        if cfg['kind'] != expected_kind:
            raise ValueError('Saved checkpoint kind disagrees with fixed method')
        state = blob['state_dict']
        if not isinstance(state, dict) or not state or any(
                not isinstance(k, str) or not torch.is_tensor(v) or
                (v.is_floating_point() and not torch.isfinite(v).all()) for k, v in state.items()):
            raise ValueError('Invalid or nonfinite state_dict')
        with torch.random.fork_rng(devices=[]):
            model = model_from_config(cfg)
        model.load_state_dict(state, strict=True)
        return model.eval()


def load_saved_inputs(saved_run: str | Path, data_path: str | Path | None = None) -> SavedInputs:
    root = Path(saved_run).resolve()
    manifest = read_json(root / 'manifest.json')
    audit = verify_artifacts(root)
    if not audit['ok']:
        raise ValueError(f'Saved run integrity failed: {audit["errors"]}')
    result = read_json(root / 'result.json')
    if result.get('schema') != 'nrr.comparison.v1':
        raise ValueError('Unsupported saved comparison schema')
    if result.get('protocol', {}).get('methods') != list(METHODS) or set(result.get('methods', {})) != set(METHODS):
        raise ValueError('Saved methods must match the five preregistered arms')
    dm = read_json(root / 'data_manifest.json')
    if dm != result.get('data'):
        raise ValueError('Saved data manifests disagree')
    if data_path is None:
        if dm.get('source') != DIGITS_SOURCE:
            raise ValueError('Original NPZ is required for this dataset; no digits substitution')
        raw = load_digits()
        arrays = {k: np.asarray(dm[k]['ids'], dtype=np.int64) for k in ('train', 'val', 'test')}
        data = Dataset((raw.images[:, None] / 16).astype(np.float32), raw.target.astype(np.int64),
                       **arrays, source=dm['source'], scope=dm['scope'])
    else:
        data = load_npz(data_path)
    actual = data.manifest()
    if any(actual[k] != dm[k] for k in ('data_sha256', 'shape', 'classes', 'train', 'val', 'test')):
        raise ValueError('Restored data or split hashes disagree with original run')
    raw_ids = result.get('calibration_ids')
    if not isinstance(raw_ids, list) or not raw_ids or any(type(v) is not int for v in raw_ids):
        raise ValueError('Invalid calibration IDs')
    ids = np.asarray(raw_ids, dtype=np.int64)
    if len(set(ids)) != len(ids) or not set(ids).issubset(set(data.train)):
        raise ValueError('calibration must contain unique original train IDs only')
    cm = read_json(root / 'calibration.json')
    if cm != {'split':'train', 'ids': raw_ids, 'sha256': digest_array(ids, data.x[ids])}:
        raise ValueError('calibration identity differs from saved source')
    checkpoints = {'teacher': root / 'teacher/model.pt', **{n: root / n / 'selected.pt' for n in METHODS}}
    required = {str(p.relative_to(root)) for p in checkpoints.values()}
    if not required.issubset(manifest['files']):
        raise ValueError('Saved manifest does not cover all trained checkpoints')
    snapshot = {n: row['sha256'] for n, row in manifest['files'].items()}
    snapshot['manifest.json'] = digest_file(root / 'manifest.json')
    return SavedInputs(root, result, data, ids, checkpoints, snapshot)


def tool_availability() -> dict:
    packages = {}
    for module, distribution in [('tensorflow','tensorflow'), ('ethosu','ethos-u-vela')]:
        available = importlib.util.find_spec(module) is not None
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            try:
                version = importlib.metadata.version('tensorflow-cpu') if module == 'tensorflow' else None
            except importlib.metadata.PackageNotFoundError:
                version = None
        packages[module] = {'discoverable':available, 'distribution_version':version}
    return {'packages':packages, 'vela_executable':shutil.which('vela'),
            'scope':'module discovery only, not proof of successful import or compilation'}


def preflight(saved_run: str | Path, out: str | Path, data_path: str | Path | None = None) -> dict:
    p = load_saved_inputs(saved_run, data_path)
    dest = reserve_output(out)
    try:
        checkpoints = {}
        for name, path in p.checkpoints.items():
            model = p.load_model(name)
            checkpoints[name] = {'source_file':str(path.relative_to(p.root)),
                                 'file_sha256':digest_file(path), 'state_sha256':state_digest(model),
                                 'config':model.config, 'parameters':sum(v.numel() for v in model.parameters())}
        p.assert_preserved()
        report = {'schema':'nrr.saved-input-preflight.v1', 'status':'inputs_verified',
                  'saved_inputs_verified':True, 'source_manifest_sha256':p.source_snapshot['manifest.json'],
                  'source_result_sha256':p.source_snapshot['result.json'], 'checkpoints':checkpoints,
                  'test_images':len(p.data.test), 'calibration_images':len(p.calibration_ids),
                  'validation_parity_ids':p.data.val[:16].tolist(),
                  'new_training_updates':0, 'compiler_executed':False, 'npu_hardware_executed':False,
                  'dependencies':tool_availability(), 'environment':environment(Path(__file__).resolve().parents[2])}
        write_json(dest / 'preflight.json', report)
        (dest / '.running').unlink()
        return report
    except Exception as exc:
        write_json(dest / 'blocked.json', {'status':'blocked','reason':str(exc),
                   'exception':type(exc).__name__,'compiler_executed':False})
        raise
