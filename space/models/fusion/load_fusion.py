"""Portable loader for the locally trained Fusion Attention CW checkpoint."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def load_fusion(root=ROOT):
    """Return (Keras classifier, class_names ordered by output ID)."""
    root = Path(root).resolve()
    info = json.loads((root / "model_info.json").read_text(encoding="utf-8"))
    path = root / info["checkpoint"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != info["checkpoint_sha256"]:
        raise ValueError("Fusion checkpoint checksum mismatch")
    mapping = json.loads((root / "label_mapping.json").read_text(encoding="utf-8"))
    canonical = json.dumps(mapping, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if (len(mapping) != 100 or sorted(mapping.values()) != list(range(100))
            or hashlib.sha256(canonical.encode()).hexdigest() != info["class_mapping_sha256_canonical"]):
        raise ValueError("Incorrect class mapping")
    sys.path.insert(0, str(root / "source"))
    import tensorflow as tf
    import fusion_vsl.models  # Registers the original custom Keras layers.
    import fusion_vsl.fusion_models
    model = tf.keras.models.load_model(path, compile=False, safe_mode=True)
    if model.output_shape != (None, 100):
        raise ValueError("Expected the original 100-class classifier")
    names = [name for name, label in sorted(mapping.items(), key=lambda item: item[1])]
    return model, names
