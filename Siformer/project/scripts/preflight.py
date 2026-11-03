"""Read-only TensorFlow GPU preflight; no optimizer step or test inference."""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import numpy as np
    import tensorflow as tf
    from vsl.common import load_config, read_json
    from vsl_tf.data import LandmarkDataset
    from vsl_tf.engine import seed_everything
    from vsl_tf.model import SiformerVSL
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    a = p.parse_args()
    cfg = load_config(a.config)
    if cfg["device"] != "cuda" or not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("Enable Kaggle GPU; TensorFlow must see a physical GPU")
    mapping = read_json(cfg["label_mapping"])
    if len(mapping) != 100 or sorted(mapping.values()) != list(range(100)):
        raise ValueError("Fixed 100-class label mapping differs")
    with (Path(__file__).resolve().parents[1] / "docs/dataset_counts.csv").open(encoding="utf-8-sig") as f:
        original = list(csv.DictReader(f))
    manifest = read_json(Path(cfg["manifest_dir"]) / "videos.json")
    for split in ("train", "validation", "test"):
        rows = [r for r in manifest if r["split"] == split]
        counts = Counter(r["label_name"] for r in rows)
        if {r["label"] for r in rows} != set(range(100)):
            raise ValueError(f"{split}: labels differ")
        for row in original:
            if counts[row["label_name"]] != int(row[split]):
                raise ValueError(f"{split}/{row['label_name']}: count differs from local")
        if cfg["extractor"]["name"] != "imported_landmark" and any(not (Path(cfg["data_root"]) / r["path"]).is_file() for r in rows):
            raise FileNotFoundError(f"{split}: missing RGB video")
        print(split, len(rows), "original videos")
    datasets = {s: LandmarkDataset(cfg, s, allow_errors=s == "test") for s in ("train", "validation", "test")}
    for split, dataset in datasets.items():
        print(split, len(dataset), "cache records; errors", len(dataset.errors))
    sample = datasets["validation"].get(0)
    repeat = datasets["validation"].get(0)
    if not np.array_equal(sample["coords"], repeat["coords"]):
        raise AssertionError("Validation preprocessing is random")
    x = np.stack([sample["coords"]])
    if x.shape != (1, cfg["frames"], 54, 2) or not np.isfinite(x).all():
        raise ValueError("Input shape or finite-value check failed")
    if sample["mask"].shape != (cfg["frames"], 54):
        raise ValueError("Mask shape invalid")
    seed_everything(cfg["seed"])
    model = SiformerVSL(cfg)
    tensor = model(tf.convert_to_tensor(x), training=False)
    if "GPU:" not in tensor.device.upper():
        raise RuntimeError(f"Siformer logits were placed on {tensor.device}, not GPU")
    logits = tensor.numpy()
    model.build_exit_heads()
    if logits.shape != (1, 100) or not np.isfinite(logits).all():
        raise ValueError("GPU model forward failed")
    print("TensorFlow GPU forward OK; preflight does not train or read test samples")


if __name__ == "__main__":
    main()
