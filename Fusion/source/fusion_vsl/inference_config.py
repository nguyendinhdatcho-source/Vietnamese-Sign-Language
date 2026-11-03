"""M11-only configuration and safe publication. No training/encoder imports."""
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile

METHODS = ("selected", "rgb", "landmark", "fusion", "fusion_mean", "late_fusion")
VERSION = "m11_external_isolated_sign_v1"


class InferenceError(ValueError):
    """Fail closed; no fabricated prediction or silent model fallback."""


def require(condition, message):
    if not condition:
        raise InferenceError("M11: " + message)


@dataclass(frozen=True)
class InferenceConfig:
    model: str = "selected"
    device: str = "cpu"
    max_file_bytes: int = 64 * 1024**2
    max_frames: int = 600
    max_duration_seconds: float = 20.0
    max_frame_pixels: int = 1920 * 1080
    max_decoded_bytes: int = 256 * 1024**2
    worker_timeout: float = 180.0
    keep_intermediates: bool = False

    def __post_init__(self):
        require(self.model in METHODS and self.device == "cpu", "only supported models / explicit CPU are allowed")
        for key in ("max_file_bytes", "max_frames", "max_frame_pixels", "max_decoded_bytes"):
            require(type(getattr(self, key)) is int and getattr(self, key) > 0, key + " must be positive integer")
        for key in ("max_duration_seconds", "worker_timeout"):
            value = getattr(self, key)
            require(type(value) in (int, float) and math.isfinite(value) and value > 0, key + " must be positive finite")
        require(type(self.keep_intermediates) is bool, "keep_intermediates must be bool")

    def to_dict(self):
        return asdict(self)


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def publish_json(path, value):
    """Atomic no-overwrite commit via same-filesystem hard link; UTF-8 JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), "output already exists: " + str(path))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # refuses existing destination, unlike os.replace
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    require(read_json(path) == value, "JSON publication/readback mismatch")
    return path
