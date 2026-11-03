"""M00: Project configuration and validation."""
import json
import hashlib
import importlib.metadata
import os
import platform
import random
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path


def load_config(path):
    with open(path, encoding='utf-8') as f:
        cfg = json.load(f)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    if not isinstance(cfg, dict):
        raise ValueError('M00: config must be a JSON object')
    mandatory = ['num_classes', 'frames', 'image_size', 'rgb_feature_dim', 'seed',
                 'landmark_feature_dim', 'expected_counts', 'manifest_csv', 'data_root',
                 'artifacts_dir', 'reports_dir', 'handoff_root']
    missing = [name for name in mandatory if name not in cfg]
    if missing:
        raise ValueError(f'M00 missing config entries: {missing}')
    for name in ['num_classes', 'frames', 'image_size', 'rgb_feature_dim', 'landmark_feature_dim']:
        if type(cfg[name]) is not int or cfg[name] <= 0:
            raise ValueError(f'M00: {name} must be a positive integer')
    if cfg['num_classes'] != 100:
        raise ValueError('M00: Olympic AI config must have 100 classes')
    if cfg['frames'] != 16 or cfg['rgb_feature_dim'] != 1280 or cfg['landmark_feature_dim'] != 300:
        raise ValueError('M00: expected input shapes T=16, RGB=1280, landmark=300')
    if cfg['image_size'] != 224:
        raise ValueError('M00: image_size must be 224')
    _validate_seed(cfg['seed'])
    if not isinstance(cfg['expected_counts'], dict) or set(cfg['expected_counts']) != {'train','valid','test'}:
        raise ValueError('M00: wrong split names')
    if any(type(v) is not int or v <= 0 for v in cfg['expected_counts'].values()):
        raise ValueError('M00: split sizes must be positive integers')
    if cfg['expected_counts'] != {'train': 3077, 'valid': 385, 'test': 404}:
        raise ValueError('M00: locked split counts must be 3077/385/404; never resplit')
    for name in ['handoff_root', 'data_root', 'manifest_csv', 'artifacts_dir', 'reports_dir']:
        if not isinstance(cfg[name], str) or not cfg[name].strip():
            raise ValueError(f'M00: {name} must be a nonempty path')


def _validate_seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('M00: seed must be an integer in [0, 2**32)')


def resolve_paths(cfg, config_path):
    """Resolve relative paths against config.json, never against the shell CWD."""
    base = Path(config_path).resolve().parent
    paths = {key: (base / cfg[key]).resolve()
             for key in ['handoff_root', 'data_root', 'manifest_csv', 'artifacts_dir', 'reports_dir']}
    for key in ['manifest_csv', 'artifacts_dir', 'reports_dir']:
        if any(paths[key].is_relative_to(paths[source]) for source in ['data_root', 'handoff_root']):
            raise ValueError(f'M00: {key} must be outside the read-only dataset')
    if paths['manifest_csv'] in (paths['artifacts_dir'], paths['reports_dir']):
        raise ValueError('M00: manifest_csv cannot be an output directory')
    return paths


def set_seed(seed, use_tensorflow=True):
    """Seed Python/NumPy and optionally TF; enable deterministic TF operations.

    PYTHONHASHSEED must be set BEFORE interpreter startup. We only report it,
    never claim that assigning it here would reseed the running interpreter.
    New np.random.default_rng instances must explicitly receive this seed.
    """
    _validate_seed(seed)
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    state = {'seed': seed, 'python': True, 'numpy': True,
             'tensorflow': 'not_requested', 'tf_op_determinism': False,
             'pythonhashseed_environment': os.environ.get('PYTHONHASHSEED'),
             'pythonhashseed_note': 'Only effective when set before interpreter startup.'}
    if use_tensorflow:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
        state.update(tensorflow=tf.__version__, tf_op_determinism=True)
    return state


def sha256_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_write_text(path, text):
    """Write a complete UTF-8 artifact, then replace the destination atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def save_run_metadata(output_dir, cfg, *, project_root=None, seed_state=None, inputs=()):
    """Create a new immutable run directory with config, versions and input/code hashes."""
    validate_config(cfg)
    root = Path(project_root or Path.cwd()).resolve()
    versions = {}
    for package in ['numpy', 'tensorflow', 'keras', 'mediapipe', 'opencv-python', 'pandas']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    git = {'commit': None, 'dirty': None, 'error': None}
    try:
        git['commit'] = subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', 'HEAD'], stderr=subprocess.PIPE,
            text=True, timeout=10).strip()
        git['dirty'] = bool(subprocess.check_output(
            ['git', '-C', str(root), 'status', '--porcelain'], stderr=subprocess.PIPE,
            text=True, timeout=10).strip())
    except (OSError, subprocess.SubprocessError) as error:
        git['error'] = 'Git commit unavailable: ' + type(error).__name__
    sources = sorted(set(root.glob('*.md')) | set(root.glob('*.json')) |
                     set(root.glob('src/**/*.py')) | set(root.glob('scripts/*.py')) |
                     set(root.glob('tests/*.py')))
    now = datetime.now(timezone.utc)
    run_id = now.strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid.uuid4().hex[:8]
    run_dir = Path(output_dir) / 'runs' / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {'schema_version': 1, 'run_id': run_id, 'created_at_utc': now.isoformat(),
                'config': cfg, 'python': platform.python_version(),
                'platform': platform.platform(), 'package_versions': versions, 'git': git,
                'seed_state': seed_state,
                'source_sha256': {p.relative_to(root).as_posix(): sha256_file(p) for p in sources},
                'input_sha256': {str(Path(p).resolve()): sha256_file(p) for p in inputs}}
    atomic_write_text(run_dir / 'config.json', json.dumps(cfg, ensure_ascii=False, indent=2) + '\n')
    path = run_dir / 'run_metadata.json'
    atomic_write_text(path, json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    return path
