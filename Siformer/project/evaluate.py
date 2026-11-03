"""Once-only final TensorFlow test evaluation from a locked validation policy."""
import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf

from vsl.common import read_json, sha256, write_csv, write_json
from vsl.metrics import export_metrics
from vsl_tf.data import LandmarkDataset
from vsl_tf.engine import code_identity, config_identity, dataset_identity, load_checkpoint, seed_everything
from vsl_tf.model import SiformerVSL


def evaluate(policy_path, output):
    policy = read_json(policy_path)
    cfg = policy["config"]
    metadata = Path(policy["checkpoint"])
    if not metadata.is_file():
        metadata = Path(policy_path).parent / metadata.name
    if policy["test_used"] or not metadata.is_file() or sha256(metadata) != policy["checkpoint_sha256"]:
        raise ValueError("Invalid or missing locked inference policy/checkpoint")
    if (config_identity(cfg) != policy["config_identity"] or
            dataset_identity(cfg) != policy["dataset_identity"] or
            code_identity() != policy["code_identity"]):
        raise ValueError("Locked config, dataset or TensorFlow source has changed")
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    seed_everything(policy["inference_seed"])
    model = SiformerVSL(cfg)
    state = load_checkpoint(metadata, model, cfg)
    if state["smoke"]:
        raise ValueError("Smoke checkpoint is not a final model")
    model.patience = int(policy["patience"])
    labels = state["labels"]
    names = {i: name for name, i in labels.items()}
    test = LandmarkDataset(cfg, "test", allow_errors=True)
    validation = LandmarkDataset(cfg, "validation")
    benchmark = tf.convert_to_tensor(validation.get(0)["coords"][None], tf.float32)
    for _ in range(cfg["evaluation"]["warmup"]):
        model(benchmark, training=False).numpy()  # Synchronizes GPU.
    durations = []
    for _ in range(cfg["evaluation"]["latency_repeats"]):
        start = time.perf_counter()
        model(benchmark, training=False).numpy()
        durations.append((time.perf_counter() - start) * 1000)
    seed_everything(policy["inference_seed"])
    model.attention_rng.reset_from_seed(cfg["seed"] + 17)
    out.mkdir(parents=True)
    write_json(out / "policy.json", policy)
    write_json(out / "labels.json", labels)
    rows, errors, exits = [], [], Counter()
    for index, meta in enumerate(test.rows):
        row = {"video_id": meta["video_id"], "path": meta["path"],
               "true_label": meta["label"], "true_name": names[meta["label"]],
               "predicted_label": None, "predicted_name": None, "confidence": None,
               "correct": False, "error": None}
        try:
            sample = test.get(index)
        except (RuntimeError, ValueError, OSError) as exc:
            row["error"] = str(exc)
            errors.append(row.copy())
        else:
            logits = model(tf.convert_to_tensor(sample["coords"][None], tf.float32), training=False)
            probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
            prediction = int(probs.argmax())
            row.update(predicted_label=prediction, predicted_name=names[prediction],
                       confidence=float(probs[prediction]), correct=prediction == meta["label"])
            exits.update([",".join(map(str, model.exit_counts[0]))])
        rows.append(row)
    if len(rows) != len(test) or len({r["video_id"] for r in rows}) != len(test):
        raise AssertionError("Expected exactly one record per test manifest video")
    write_csv(out / "predictions.csv", rows)
    write_json(out / "errors.json", errors)
    metrics = export_metrics(rows, labels, out)
    write_json(out / "model_info.json", {
        "parameters_total": model.count_params(),
        "parameters_trainable": sum(int(tf.size(v)) for v in model.trainable_variables),
        "mode": policy["mode"], "patience": model.patience,
        "encoder_exit_layer_counts_left_right_body": dict(exits),
        "latency": {"scope": "TensorFlow model only; batch=1; validation clip; excludes I/O/extraction",
                    "device": cfg["device"], "warmup": cfg["evaluation"]["warmup"],
                    "repeats": len(durations), "mean_ms": float(np.mean(durations)),
                    "median_ms": float(np.median(durations)),
                    "p95_ms": float(np.percentile(durations, 95))},
        "end_to_end_latency": "not measured; offline extraction is not inference latency",
        "tensorflow_version": tf.__version__,
        "physical_gpus": [str(g) for g in tf.config.list_physical_devices("GPU")]})
    print(metrics)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    evaluate(a.policy, a.output)


if __name__ == "__main__":
    main()
