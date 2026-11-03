import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "979a14ed15ed0f20afd77d447ad23c4f4107a2c3"


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if cfg["num_classes"] != 100 or cfg["frames"] < 2:
        raise ValueError("This experiment requires 100 classes and frames >= 2")
    if cfg["sampling"] != "uniform_floor":
        raise ValueError("Unknown sampling rule")
    if cfg["model"]["pretrained"] is not None:
        raise ValueError("No compatible pretrained checkpoint has been verified. Use scratch/resume.")
    if cfg["model"]["inference"] != "static" or cfg["model"]["patience"] != 0:
        raise ValueError("Train config must use static inference. Use calibrate.py for a locked adaptive policy.")
    if cfg["model"]["encoder_layers"] < 1 or cfg["model"]["decoder_layers"] < 1:
        raise ValueError("Encoder/decoder depth must be positive")
    if cfg["train"]["batch_size"] < 1 or cfg["train"]["epochs"] < 1:
        raise ValueError("Batch size/epochs must be positive")
    if cfg["evaluation"]["warmup"] < 1 or cfg["evaluation"]["latency_repeats"] < 1:
        raise ValueError("Latency needs warmup and repeated timings")
    return cfg


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    import os
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    # Atomic metadata update; epoch checkpoint files remain immutable.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as f:
        f.write(payload)
        temporary = f.name
    os.replace(temporary, path)


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def cache_signature(cfg):
    return digest({k: cfg[k] for k in
                   ("extractor", "augmentation", "offline_variants", "rgb_resize", "rectification")})


def vendor_imports():
    """Called explicitly by model/adapter construction, never on package import."""
    import sys
    base = ROOT / "vendor" / "siformer_official"
    for p in (base, base / "rectifications"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
