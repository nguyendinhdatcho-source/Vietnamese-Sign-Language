"""TensorFlow training, integrity-checked checkpointing, and resume primitives."""
import copy
import ast
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from vsl.common import ROOT, digest, read_json, sha256, write_json


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def require_configured_device(cfg, output):
    if cfg["device"] == "cuda" and "GPU:" not in output.device.upper():
        raise RuntimeError(f"TensorFlow placed Siformer logits on {output.device}, not GPU")


def code_identity():
    files = list((ROOT / "vsl_tf").glob("*.py"))
    files += [ROOT / "vsl" / name for name in ("adapter.py", "augmentation.py", "common.py", "metrics.py")]
    files += [ROOT / name for name in ("prepare_data.py", "train.py", "calibrate.py", "evaluate.py")]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(files)}


def dataset_identity(cfg):
    return {name: sha256(Path(cfg["manifest_dir"]) / name) for name in ("videos.json", "labels.json")} | {
        "cache_index": sha256(Path(cfg["cache_dir"]) / "index.json"),
        "cache_provenance": sha256(Path(cfg["cache_dir"]) / "provenance.json")}


def config_identity(cfg):
    value = copy.deepcopy(cfg)
    for name in ("run_dir", "data_root", "manifest_dir", "cache_dir", "label_mapping"):
        value.pop(name, None)
    value["train"].pop("epochs")
    return digest(value)


def make_optimizer(cfg):
    options = cfg["train"]
    if options["optimizer"] != "AdamW" or options["scheduler"] != "MultiStepLR":
        raise ValueError("This migration supports AdamW and MultiStepLR only")
    kwargs = dict(learning_rate=options["lr"], weight_decay=options["weight_decay"],
                  beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    # Keras applies global_clipnorm inside apply_gradients, after loss unscaling.
    if options["gradient_clip"] is not None:
        kwargs["global_clipnorm"] = options["gradient_clip"]
    base = tf.keras.optimizers.AdamW(**kwargs)
    return tf.keras.mixed_precision.LossScaleOptimizer(base) if options["amp"] else base


def learning_rate(optimizer):
    return optimizer.inner_optimizer.learning_rate if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer) else optimizer.learning_rate


def next_epoch_lr(optimizer, cfg, epoch):
    if epoch + 1 in cfg["train"]["milestones"]:
        lr = learning_rate(optimizer)
        lr.assign(lr * cfg["train"]["gamma"])


def macro_f1(truth, predicted):
    matrix = np.zeros((100, 100), np.int64)
    np.add.at(matrix, (np.asarray(truth), np.asarray(predicted)), 1)
    tp = np.diag(matrix).astype(np.float64)
    support, called = matrix.sum(1), matrix.sum(0)
    p = np.divide(tp, called, out=np.zeros(100), where=called != 0)
    r = np.divide(tp, support, out=np.zeros(100), where=support != 0)
    return float(np.divide(2 * p * r, p + r, out=np.zeros(100), where=p + r != 0).mean())


def run_epoch(model, dataset, cfg, rng, optimizer=None, max_batches=None):
    training = optimizer is not None
    total_loss, truth, predictions = 0.0, [], []

    if training:
        @tf.function(reduce_retracing=True)
        def step(x, y):
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=y, logits=tf.cast(logits, tf.float32)))
                scaled = optimizer.scale_loss(loss) if cfg["train"]["amp"] else loss
            variables = model.trainable_variables
            grads = tape.gradient(scaled, variables)
            if any(g is None for g in grads):
                raise RuntimeError("Some model variables have no gradients")
            if not cfg["train"]["amp"]:
                for gradient in grads:
                    tf.debugging.assert_all_finite(gradient, "Nonfinite float32 gradient")
            # AMP: LossScaleOptimizer unscales, skips overflowed steps, adjusts loss scale.
            optimizer.apply_gradients(zip(grads, variables))
            return loss, tf.argmax(logits, axis=-1, output_type=tf.int64)
    else:
        @tf.function(reduce_retracing=True)
        def step(x, y):
            logits = model(x, training=False)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y, logits=tf.cast(logits, tf.float32)))
            return loss, tf.argmax(logits, axis=-1, output_type=tf.int64)

    for batch_index, batch in enumerate(dataset.batches(rng)):
        if max_batches is not None and batch_index >= max_batches:
            break
        context = f"split={dataset.split} batch={batch_index} videos={batch['video_id']}"
        x = tf.convert_to_tensor(batch["coords"], tf.float32)
        y = tf.convert_to_tensor(batch["label"], tf.int64)
        if not bool(tf.reduce_all((y >= 0) & (y < 100))):
            raise ValueError(f"Label out of range: {context}")
        tf.debugging.assert_all_finite(x, f"Nonfinite input: {context}")
        try:
            loss, pred = step(x, y)
            tf.debugging.assert_all_finite(loss, f"Nonfinite loss: {context}")
        except Exception as exc:
            raise RuntimeError(f"TensorFlow step failed: {context}: {exc}") from exc
        total_loss += float(loss.numpy()) * len(batch["label"])
        truth.extend(batch["label"].tolist())
        predictions.extend(pred.numpy().tolist())
    if not truth:
        raise RuntimeError(f"No samples processed: {dataset.split}")
    return {"loss": total_loss / len(truth),
            "accuracy": float(np.mean(np.asarray(truth) == predictions)),
            "macro_f1": macro_f1(truth, predictions), "samples": len(truth)}

def make_checkpoint(model, optimizer=None):
    epoch = tf.Variable(-1, dtype=tf.int64, trainable=False)
    best_accuracy = tf.Variable(-1.0, dtype=tf.float64, trainable=False)
    best_negative_loss = tf.Variable(float("-inf"), dtype=tf.float64, trainable=False)
    bad_epochs = tf.Variable(0, dtype=tf.int64, trainable=False)
    parts = dict(model=model, attention_rng=model.attention_rng, epoch=epoch,
                 best_accuracy=best_accuracy, best_negative_loss=best_negative_loss, bad_epochs=bad_epochs)
    if optimizer is not None:
        parts["optimizer"] = optimizer
    return tf.train.Checkpoint(**parts)


def _checkpoint_files(prefix):
    prefix = Path(prefix)
    return [Path(str(prefix) + ".index"), *sorted(prefix.parent.glob(prefix.name + ".data-*"))]


def resolve_checkpoint(pointer):
    pointer = Path(pointer)
    if pointer.suffix == ".pt":
        raise ValueError("PyTorch .pt checkpoints cannot be loaded as TensorFlow checkpoints")
    if pointer.suffix == ".json":
        reference = read_json(pointer)
        prefix = pointer.parent / Path(reference["checkpoint"]).name
        for filename, expected in reference["files"].items():
            path = pointer.parent / filename
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"Checkpoint file missing/hash mismatch: {path}")
    else:
        prefix = pointer
    files = _checkpoint_files(prefix)
    if len(files) < 2 or not all(p.is_file() for p in files):
        raise FileNotFoundError(f"Incomplete TensorFlow checkpoint: {prefix}")
    return prefix


def save_checkpoint(run, trackable, cfg, epoch, best_score, bad_epochs, history, rng, smoke):
    run = Path(run)
    prefix = run / f"epoch_{epoch:04d}.ckpt"
    if any(p.exists() for p in _checkpoint_files(prefix)) or (run / f"epoch_{epoch:04d}.json").exists():
        raise FileExistsError(prefix)
    trackable.epoch.assign(epoch)
    trackable.best_accuracy.assign(best_score[0])
    trackable.best_negative_loss.assign(best_score[1])
    trackable.bad_epochs.assign(bad_epochs)
    prefix = Path(trackable.save(str(prefix)))
    files = {p.name: sha256(p) for p in _checkpoint_files(prefix)}
    sidecar = {"checkpoint": prefix.name, "files": files, "epoch": epoch,
               "best_score": list(best_score), "bad_epochs": bad_epochs,
               "history": history, "numpy_rng": rng.bit_generator.state,
               "keras_dropout_rng": [v.numpy().tolist() for v in trackable.model.non_trainable_variables
                                       if "seed_generator" in v.path],
               "python_rng": repr(random.getstate()), "config": cfg,
               "config_identity": config_identity(cfg), "dataset_identity": dataset_identity(cfg),
               "code_identity": code_identity(),
               "labels": read_json(Path(cfg["manifest_dir"]) / "labels.json"),
               "tensorflow_version": tf.__version__, "smoke": smoke,
               "method": "Siformer-VSL-MediaPipe-2D TensorFlow compatibility adaptation"}
    metadata = run / f"epoch_{epoch:04d}.json"
    write_json(metadata, sidecar)
    return {"checkpoint": metadata.name, "sha256": sha256(metadata)}


def resolve_pointer(pointer):
    pointer = Path(pointer)
    ref = read_json(pointer)
    metadata = pointer.parent / Path(ref["checkpoint"]).name
    if not metadata.is_file() or sha256(metadata) != ref["sha256"]:
        raise ValueError(f"Checkpoint metadata missing/hash mismatch: {pointer}")
    return metadata


def load_checkpoint(path, model, cfg, optimizer=None, rng=None, strict=True):
    path = Path(path)
    metadata = resolve_pointer(path) if path.name in ("best.json", "last.json") else path
    state = read_json(metadata)
    if state["config_identity"] != config_identity(cfg) or state["dataset_identity"] != dataset_identity(cfg):
        raise ValueError("Configuration or dataset/cache identity differs from checkpoint")
    if strict and state["code_identity"] != code_identity():
        raise ValueError("Source-code identity differs from checkpoint")
    if state["labels"] != read_json(Path(cfg["manifest_dir"]) / "labels.json"):
        raise ValueError("Label mapping differs from checkpoint")
    if optimizer is not None and state["tensorflow_version"] != tf.__version__:
        raise ValueError("Exact optimizer resume requires the saved TensorFlow version")
    # Build all model/optimizer variables before restore, including frozen early-exit heads.
    probe = model(tf.zeros((1, cfg["frames"], 54, 2)), training=False)
    require_configured_device(cfg, probe)
    model.build_exit_heads()
    if optimizer is not None:
        optimizer.build(model.trainable_variables)
    ckpt = make_checkpoint(model, optimizer)
    status = ckpt.restore(str(resolve_checkpoint(metadata.parent / state["checkpoint"])))
    status.assert_existing_objects_matched()
    if optimizer is None:
        # Inference intentionally does not create optimizer slots; every model
        # object was matched above, while saved optimizer objects are ignored.
        status.expect_partial()
    dropout_rng = [v for v in model.non_trainable_variables if "seed_generator" in v.path]
    if len(dropout_rng) != len(state["keras_dropout_rng"]):
        raise ValueError("Keras dropout RNG topology differs from saved checkpoint")
    for variable, value in zip(dropout_rng, state["keras_dropout_rng"]):
        variable.assign(value)
    if optimizer is not None:
        status.assert_consumed()
        if rng is None:
            raise ValueError("Resume requires NumPy sampling RNG")
        rng.bit_generator.state = state["numpy_rng"]
        random.setstate(ast.literal_eval(state["python_rng"]))
    return state


def policy_checkpoint_hash(metadata):
    metadata = Path(metadata)
    return sha256(metadata)
