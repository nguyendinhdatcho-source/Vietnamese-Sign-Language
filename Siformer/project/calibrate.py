"""Validation-only static/early-exit inference policy for TensorFlow."""
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from vsl.common import load_config, write_json
from vsl_tf.data import LandmarkDataset
from vsl_tf.engine import (code_identity, config_identity, dataset_identity, load_checkpoint,
                           policy_checkpoint_hash, resolve_pointer, run_epoch, seed_everything)
from vsl_tf.model import SiformerVSL


def calibrate(cfg, checkpoint, output, adaptive=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.name != "best.json":
        raise ValueError("Only validation-selected best.json may be locked for test")
    seed_everything(cfg["seed"])
    model = SiformerVSL(cfg)
    state = load_checkpoint(checkpoint, model, cfg)
    if state["smoke"]:
        raise ValueError("Smoke checkpoint cannot be used for final evaluation")
    candidates = cfg["evaluation"]["calibration_patiences"] if adaptive else [0]
    if not candidates or 0 not in candidates or any(p < 0 for p in candidates):
        raise ValueError("Patience candidates must include 0 and be nonnegative")
    validation = LandmarkDataset(cfg, "validation")
    results = []
    for patience in candidates:
        seed_everything(cfg["seed"] + 10)
        model.attention_rng.reset_from_seed(cfg["seed"] + 17)
        model.patience = int(patience)
        metrics = run_epoch(model, validation, cfg, np.random.default_rng(cfg["seed"] + 11))
        results.append({"patience": int(patience), **metrics})
    best = max(results, key=lambda r: (r["accuracy"], -r["loss"], r["patience"] == 0))
    metadata = resolve_pointer(checkpoint)
    write_json(output, {"checkpoint": str(metadata), "checkpoint_sha256": policy_checkpoint_hash(metadata),
                        "config": cfg, "config_identity": config_identity(cfg),
                        "dataset_identity": dataset_identity(cfg), "code_identity": code_identity(),
                        "mode": "static" if best["patience"] == 0 else "encoder_patience",
                        "patience": best["patience"], "validation_candidates": results,
                        "selection": "validation accuracy, then validation loss, then static",
                        "test_used": False, "inference_seed": cfg["seed"] + 20})
    print(f"Locked policy: {output}; patience={best['patience']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train.json")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--adaptive", action="store_true")
    a = p.parse_args()
    calibrate(load_config(a.config), a.checkpoint, a.output, a.adaptive)


if __name__ == "__main__":
    main()
