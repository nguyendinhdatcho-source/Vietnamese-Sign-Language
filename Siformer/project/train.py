"""TensorFlow Siformer training entry point."""
import argparse
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf

from vsl.common import load_config, read_json, write_csv, write_json, sha256
from vsl_tf.data import LandmarkDataset, sampling_rows
from vsl_tf.engine import (load_checkpoint, learning_rate, make_checkpoint, make_optimizer,
                           next_epoch_lr, require_configured_device, resolve_pointer, run_epoch,
                           save_checkpoint, seed_everything)
from vsl_tf.model import SiformerVSL


def train(cfg, resume=None, smoke=False):
    out = Path(cfg["run_dir"])
    if out.exists():
        raise FileExistsError(f"Use a NEW run_dir, including resumed branches: {out}")
    if cfg["device"] == "cuda" and not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("TensorFlow sees no GPU; enable Kaggle GPU or set device=cpu")
    if cfg["train"]["amp"] and cfg["device"] != "cuda":
        raise ValueError("AMP is configured for GPU only")
    tf.keras.mixed_precision.set_global_policy("mixed_float16" if cfg["train"]["amp"] else "float32")
    seed_everything(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    training_set = LandmarkDataset(cfg, "train")
    validation_set = LandmarkDataset(cfg, "validation")
    if set(training_set.labels.tolist()) != set(range(100)):
        raise ValueError("Training split must contain all 100 labels")
    model = SiformerVSL(cfg)
    probe = model(tf.zeros((1, cfg["frames"], 54, 2)), training=False)
    require_configured_device(cfg, probe)
    model.build_exit_heads()
    optimizer = make_optimizer(cfg)
    optimizer.build(model.trainable_variables)
    checkpoint = make_checkpoint(model, optimizer)
    start, best, bad_epochs, history = 0, (-1.0, float("-inf")), 0, []
    inherited_best = None
    if resume:
        if Path(resume).name != "last.json":
            raise ValueError("Resume requires last.json and matching best.json")
        state = load_checkpoint(resume, model, cfg, optimizer, rng)
        if state["smoke"] != smoke:
            raise ValueError("Cannot resume smoke as full training")
        start, best = state["epoch"] + 1, tuple(state["best_score"])
        bad_epochs, history = state["bad_epochs"], state["history"]
        inherited_best = resolve_pointer(Path(resume).parent / "best.json")
    if start >= cfg["train"]["epochs"]:
        raise ValueError("No remaining epochs; increase train.epochs")
    out.mkdir(parents=True)
    write_json(out / "config.json", cfg)
    write_json(out / "labels.json", read_json(Path(cfg["manifest_dir"]) / "labels.json"))
    weights, counts = sampling_rows(training_set)
    write_csv(out / "sampling_weights.csv", weights)
    write_csv(out / "train_class_counts.csv", counts)
    if inherited_best:
        best_state = read_json(inherited_best)
        for name in [inherited_best.name, *best_state["files"]]:
            shutil.copy2(inherited_best.parent / name, out / name)
        write_json(out / "best.json", {"checkpoint": inherited_best.name, "sha256": sha256(out / inherited_best.name)})
    stop = min(start + 1, cfg["train"]["epochs"]) if smoke else cfg["train"]["epochs"]
    for epoch in range(start, stop):
        lr = float(learning_rate(optimizer).numpy())
        training = run_epoch(model, training_set, cfg, rng, optimizer, 1 if smoke else None)
        validation = run_epoch(model, validation_set, cfg, rng, max_batches=1 if smoke else None)
        score = (validation["accuracy"], -validation["loss"])
        improved = score > best
        best, bad_epochs = (score, 0) if improved else (best, bad_epochs + 1)
        history.append({"epoch": epoch, "lr": lr, "smoke": smoke,
                        **{"train_" + k: v for k, v in training.items()},
                        **{"validation_" + k: v for k, v in validation.items()}})
        next_epoch_lr(optimizer, cfg, epoch)
        pointer = save_checkpoint(out, checkpoint, cfg, epoch, best, bad_epochs, history, rng, smoke)
        write_json(out / "last.json", pointer)
        if improved:
            write_json(out / "best.json", pointer)
        write_csv(out / "history.csv", history)
        print(history[-1], flush=True)
        patience = cfg["train"]["early_stopping_patience"]
        if patience is not None and bad_epochs >= patience:
            break
    print(f"{'SMOKE ONLY' if smoke else 'Training complete'}: {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train.json")
    p.add_argument("--resume")
    p.add_argument("--run-dir")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.run_dir:
        cfg["run_dir"] = args.run_dir
    train(cfg, args.resume, args.smoke)


if __name__ == "__main__":
    main()
