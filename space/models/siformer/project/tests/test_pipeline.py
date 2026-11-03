"""CPU equivalence and TensorFlow training-state tests; GPU validation is separate."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from vsl.augmentation import temporal_sample
from vsl.adapter import normalize
from vsl.common import write_json, sha256, cache_signature
from vsl_tf.data import LandmarkDataset
from vsl_tf.engine import (load_checkpoint, make_checkpoint, make_optimizer, run_epoch,
                           save_checkpoint, seed_everything, next_epoch_lr, learning_rate)
from vsl_tf.model import SiformerVSL, ProbSparseAttention


ROOT = Path(__file__).resolve().parents[1]


def cfg(frames=8):
    import json
    result = json.loads((ROOT / "configs/train.json").read_text(encoding="utf-8"))
    result["frames"] = frames
    result["model"]["encoder_layers"] = 1
    result["model"]["decoder_layers"] = 1
    result["train"]["batch_size"] = 2
    result["train"]["workers"] = 0
    return result


def dense(layer, weights, name):
    layer.set_weights([weights["weight::" + name + ".weight"].T,
                       weights["weight::" + name + ".bias"]])


def norm(layer, weights, name):
    layer.set_weights([weights["weight::" + name + ".weight"],
                       weights["weight::" + name + ".bias"]])


def map_reference(model, weights):
    base = "core.transformer."
    for i, name in enumerate(("l_hand", "r_hand", "body")):
        model.positional[i].assign(weights[f"weight::core.{name}_embedding"][:, 0, :])
        encoder = model.encoders[i]
        prefix = base + name + "_encoder."
        norm(encoder.final_norm, weights, prefix + "norm")
        for j, layer in enumerate(encoder.layers_list):
            source = prefix + f"layers.{j}."
            for attr, target in (("query_projection", layer.attention.query),
                                 ("key_projection", layer.attention.key),
                                 ("value_projection", layer.attention.value),
                                 ("out_projection", layer.attention.output_projection)):
                dense(target, weights, source + "self_attn." + attr)
            for attr in ("linear1", "linear2"):
                dense(getattr(layer, attr), weights, source + attr)
            for attr in ("norm1", "norm2"):
                norm(getattr(layer, attr), weights, source + attr)
            dense(encoder.inner_classifiers[j], weights, prefix + f"inner_classifiers.{j}")
            dense(encoder.projections[j], weights, prefix + f"projections.{j}")
    model.class_query.assign(weights["weight::core.class_query"].transpose(1, 0, 2))
    for j, layer in enumerate(model.decoders):
        source = base + f"decoder.layers.{j}."
        full_w = weights["weight::" + source + "multihead_attn.in_proj_weight"]
        full_b = weights["weight::" + source + "multihead_attn.in_proj_bias"]
        for i, attr in enumerate(("query", "key", "value")):
            getattr(layer.cross_attention, attr).set_weights([full_w[i*108:(i+1)*108].T,
                                                             full_b[i*108:(i+1)*108]])
        dense(layer.cross_attention.projection, weights, source + "multihead_attn.out_proj")
        for attr in ("linear1", "linear2"):
            dense(getattr(layer, attr), weights, source + attr)
        for attr in ("norm1", "norm2", "norm3"):
            norm(getattr(layer, attr), weights, source + attr)
    norm(model.decoder_norm, weights, base + "decoder.norm")
    dense(model.head, weights, "core.projection")


class TensorFlowMigrationTests(unittest.TestCase):
    def test_encoder_early_exit_is_inference_only(self):
        options = cfg()
        options["model"]["encoder_layers"] = 3
        model = SiformerVSL(options)
        x = tf.zeros((2, 8, 54, 2))
        model.patience = 1
        self.assertEqual(model(x, training=False).shape, (2, 100))
        self.assertEqual(len(model.exit_counts), 2)
        self.assertTrue(all(len(row) == 3 and all(1 <= v <= 3 for v in row) for row in model.exit_counts))
        model(x, training=True)
        self.assertTrue(all(row == [3, 3, 3] for row in model.exit_counts))

    def test_smoke_train_and_resume_entrypoint(self):
        from train import train
        options = cfg()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("manifest", "cache"):
                (root / name).mkdir()
            options.update(manifest_dir=str(root / "manifest"), cache_dir=str(root / "cache"),
                           label_mapping=str(root / "labels.json"), run_dir=str(root / "first_run"))
            mapping = {str(i): i for i in range(100)}
            write_json(root / "labels.json", mapping)
            write_json(root / "manifest/labels.json", mapping)
            coords = np.full((10, 54, 2), 0.5, np.float32)
            mask = np.ones((10, 54), bool)
            manifest, index = [], []
            for split, labels in (("train", range(100)), ("validation", range(2)), ("test", range(2))):
                for label in labels:
                    name = f"{split}_{label}"
                    np.savez(root / "cache" / f"{name}.npz", schema="mp_holistic_xy_up_left21_right21_body12_v1",
                             coords=coords, mask=mask)
                    manifest.append({"split": split, "video_id": name, "label": label})
                    index.append({"split": split, "video_id": name, "label": label, "variant": 0,
                                  "status": "ok", "schema": "mp_holistic_xy_up_left21_right21_body12_v1",
                                  "augmentation": None, "cache_path": name + ".npz", "path": name})
            write_json(root / "manifest/videos.json", manifest)
            write_json(root / "cache/index.json", index)
            write_json(root / "cache/provenance.json", {"complete": True, "cache_signature": cache_signature(options),
                       "manifest_sha256": sha256(root / "manifest/videos.json"),
                       "labels_sha256": sha256(root / "manifest/labels.json")})
            train(options, smoke=True)
            self.assertTrue((root / "first_run/best.json").is_file())
            self.assertTrue((root / "first_run/last.json").is_file())
            options["train"]["epochs"] = 2
            options["run_dir"] = str(root / "second_run")
            train(options, resume=root / "first_run/last.json", smoke=True)
            self.assertTrue((root / "second_run/best.json").is_file())
            self.assertTrue((root / "second_run/epoch_0001.json").is_file())
            from scripts.kaggle_bundle import build_bundle
            import zipfile
            write_json(root / "runtime_config.json", options)
            archive = root / "bundle.zip"
            build_bundle(root / "runtime_config.json", archive)
            with zipfile.ZipFile(archive) as bundle:
                names = set(bundle.namelist())
            self.assertIn("bundle.json", names)
            self.assertIn("project/vsl_tf/model.py", names)
            self.assertIn("run/best.json", names)
            self.assertIn("run/last.json", names)
            self.assertTrue(any(name.endswith(".index") for name in names))
            # One actual (non-smoke) synthetic epoch, validation policy, and
            # once-only test export. This is not an accuracy experiment.
            from calibrate import calibrate
            from evaluate import evaluate
            options["run_dir"] = str(root / "synthetic_full")
            options["train"]["epochs"] = 1
            options["train"]["batch_size"] = 100
            options["evaluation"]["warmup"] = 1
            options["evaluation"]["latency_repeats"] = 2
            train(options, smoke=False)
            policy = root / "synthetic_full/policy.json"
            calibrate(options, root / "synthetic_full/best.json", policy)
            evaluate(policy, root / "synthetic_full/test")
            import json
            metrics = json.loads((root / "synthetic_full/test/metrics.json").read_text())
            report = json.loads((root / "synthetic_full/test/classification_report.json").read_text())
            self.assertEqual(metrics["total_videos"], 2)
            self.assertEqual(len(report), 100)
            self.assertTrue((root / "synthetic_full/test/predictions.csv").is_file())

    def test_controlled_train_validation_epoch(self):
        options = cfg()
        seed_everything(379)
        class TinyDataset:
            def __init__(self, split):
                self.split = split
            def batches(self, rng):
                yield {"coords": np.zeros((2, 8, 54, 2), np.float32),
                       "label": np.array([1, 2], np.int64), "video_id": ["a", "b"]}
        model = SiformerVSL(options)
        model(tf.zeros((1, 8, 54, 2)))
        optimizer = make_optimizer(options)
        rng = np.random.default_rng(379)
        training = run_epoch(model, TinyDataset("train"), options, rng, optimizer)
        iteration = int(optimizer.iterations.numpy())
        before_validation = model.head.kernel.numpy().copy()
        validation = run_epoch(model, TinyDataset("validation"), options, rng)
        self.assertEqual((training["samples"], validation["samples"]), (2, 2))
        self.assertTrue(all(np.isfinite(v) for v in (training["loss"], validation["loss"])))
        self.assertEqual(int(optimizer.iterations.numpy()), iteration)
        np.testing.assert_array_equal(before_validation, model.head.kernel.numpy())

    def test_architecture_and_gradients(self):
        options = cfg()
        seed_everything(379)
        model = SiformerVSL(options)
        x = tf.random.uniform((2, 8, 54, 2))
        model(x)
        model.build_exit_heads()
        self.assertEqual((len(model.encoders), len(model.decoders)), (3, 1))
        self.assertEqual([e.layers_list[0].attention.heads for e in model.encoders], [3, 3, 2])
        self.assertEqual(model(x).shape, (2, 100))
        self.assertEqual(sum(int(tf.size(v)) for v in model.trainable_variables), 970464)
        frozen = [v.numpy().copy() for v in model.non_trainable_variables if "inner_" in v.path or "projection_" in v.path]
        optimizer = make_optimizer(options)
        optimizer.build(model.trainable_variables)
        before = model.head.kernel.numpy().copy()
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[1, 2], logits=model(x, training=True)))
        gradients = tape.gradient(loss, model.trainable_variables)
        self.assertTrue(np.isfinite(loss.numpy()))
        self.assertTrue(all(g is not None and np.isfinite(g.numpy()).all() for g in gradients))
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        self.assertFalse(np.array_equal(before, model.head.kernel.numpy()))
        current = [v.numpy() for v in model.non_trainable_variables if "inner_" in v.path or "projection_" in v.path]
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(frozen, current)))

    def test_data_sampler_and_split_rules(self):
        options = cfg()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            options.update(cache_dir=str(root / "cache"), manifest_dir=str(root / "manifest"),
                           label_mapping=str(root / "mapping.json"))
            (root / "cache").mkdir()
            (root / "manifest").mkdir()
            mapping = {str(i): i for i in range(100)}
            write_json(root / "mapping.json", mapping)
            write_json(root / "manifest/labels.json", mapping)
            coords = np.broadcast_to(np.arange(12, dtype=np.float32)[:, None, None], (12, 54, 2)).copy() / 20
            mask = np.ones((12, 54), bool)
            mask[2, 0] = False
            manifest, index = [], []
            for split in ("train", "validation", "test"):
                for label in (0, 1):
                    name = f"{split}_{label}"
                    np.savez(root / "cache" / f"{name}.npz", schema="mp_holistic_xy_up_left21_right21_body12_v1", coords=coords, mask=mask)
                    manifest.append({"split": split, "video_id": name, "label": label})
                    index.append({"split": split, "video_id": name, "label": label, "variant": 0,
                                  "status": "ok", "schema": "mp_holistic_xy_up_left21_right21_body12_v1",
                                  "augmentation": None, "cache_path": name + ".npz", "path": name})
            write_json(root / "manifest/videos.json", manifest)
            write_json(root / "cache/index.json", index)
            write_json(root / "cache/provenance.json", {"complete": True, "cache_signature": cache_signature(options),
                       "manifest_sha256": sha256(root / "manifest/videos.json"),
                       "labels_sha256": sha256(root / "manifest/labels.json")})
            validation = LandmarkDataset(options, "validation")
            first, again = validation.get(0), validation.get(0)
            expected, expected_mask, indices = temporal_sample(coords, mask, 8, 1)
            np.testing.assert_array_equal(first["indices"], indices)
            np.testing.assert_array_equal(first["mask"], expected_mask)
            np.testing.assert_array_equal(first["coords"], normalize(expected, expected_mask))
            np.testing.assert_array_equal(first["coords"], again["coords"])
            test = LandmarkDataset(options, "test")
            np.testing.assert_array_equal(test.get(0)["coords"], test.get(0)["coords"])
            with self.assertRaises(RuntimeError):
                validation.get(0, 0.9)
            with self.assertRaises(RuntimeError):
                test.get(0, 1.1)
            train = LandmarkDataset(options, "train")
            rng = np.random.default_rng(379)
            batch = next(train.batches(rng))
            self.assertEqual(batch["coords"].shape, (2, 8, 54, 2))
            self.assertTrue(np.isfinite(batch["coords"]).all())
            self.assertTrue(((batch["label"] >= 0) & (batch["label"] < 100)).all())
            slow = temporal_sample(coords, mask, 8, .9)[2]
            fast = temporal_sample(coords, mask, 8, 1.1)[2]
            self.assertFalse(np.array_equal(slow, fast))

    def test_checkpoint_optimizer_resume(self):
        options = cfg()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("manifest", "cache", "run"):
                (root / name).mkdir()
            options.update(manifest_dir=str(root / "manifest"), cache_dir=str(root / "cache"))
            options["train"]["milestones"] = [1]
            for name in ("videos.json", "labels.json"):
                write_json(root / "manifest" / name, {"x": 1})
            for name in ("index.json", "provenance.json"):
                write_json(root / "cache" / name, {"x": 1})
            rng = np.random.default_rng(42)
            model = SiformerVSL(options)
            x = tf.zeros((1, 8, 54, 2))
            model(x)
            model.build_exit_heads()
            optimizer = make_optimizer(options)
            optimizer.build(model.trainable_variables)
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[1], logits=model(x, training=True)))
            optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
            next_epoch_lr(optimizer, options, 0)
            self.assertAlmostEqual(float(learning_rate(optimizer)), 1e-5, places=8)
            saved_rng = rng.bit_generator.state
            expected_random = rng.random()
            rng.bit_generator.state = saved_rng
            pointer = save_checkpoint(root / "run", make_checkpoint(model, optimizer), options, 0,
                                      (0.5, -1.0), 0, [{"epoch": 0}], rng, False)
            write_json(root / "run/last.json", pointer)
            expected_next_training_logits = model(x, training=True).numpy()
            expected_next_attention_random = model.attention_rng.uniform((3,), dtype=tf.float32).numpy()
            expected_head = model.head.kernel.numpy().copy()
            new_model = SiformerVSL(options)
            new_optimizer = make_optimizer(options)
            restored = load_checkpoint(root / "run/last.json", new_model, options, new_optimizer, rng)
            self.assertEqual(restored["epoch"], 0)
            np.testing.assert_array_equal(expected_head, new_model.head.kernel.numpy())
            self.assertEqual(int(new_optimizer.iterations.numpy()), 1)
            self.assertAlmostEqual(float(learning_rate(new_optimizer)), 1e-5, places=8)
            self.assertEqual(rng.random(), expected_random)
            np.testing.assert_array_equal(expected_next_training_logits, new_model(x, training=True).numpy())
            np.testing.assert_array_equal(expected_next_attention_random,
                                          new_model.attention_rng.uniform((3,), dtype=tf.float32).numpy())

    def test_reference_forward(self):
        reference = ROOT / "tests/fixtures/pytorch_reference_small.npz"
        self.assertTrue(reference.is_file(), "Frozen numerical reference fixture is missing")
        with np.load(reference, allow_pickle=False) as data:
            weights = {k: data[k] for k in data.files}
        model = SiformerVSL(cfg())
        model(tf.zeros((1, 8, 54, 2)))
        model.build_exit_heads()
        map_reference(model, weights)
        actual = model(weights["input"], training=False).numpy()
        left = tf.reshape(weights["input"][:, :, :21], [1, 8, 42]) + model.positional[0][None]
        left_actual, _ = model.encoders[0](left, training=False)
        np.testing.assert_allclose(left_actual.numpy(), weights["left_memory"], rtol=2e-4, atol=2e-4)
        np.testing.assert_allclose(actual, weights["logits"], rtol=2e-4, atol=2e-4)
        print("max_abs_logit_diff", float(np.max(np.abs(actual - weights["logits"]))))

        attention = ProbSparseAttention(42, 3, tf.random.Generator.from_seed(1))
        attention(tf.zeros((1, 32, 42)))
        for attr, source in (("query", "query_projection"), ("key", "key_projection"),
                             ("value", "value_projection"), ("output_projection", "out_projection")):
            dense(getattr(attention, attr), weights, "sparse::" + source)
        indices = np.tile(np.arange(20, dtype=np.int32), (32, 1))
        sparse_actual = attention(weights["sparse_input"], sample_indices=indices).numpy()
        np.testing.assert_allclose(sparse_actual, weights["sparse_output"], rtol=2e-4, atol=2e-4)
        print("max_abs_sparse_diff", float(np.max(np.abs(sparse_actual - weights["sparse_output"]))))


if __name__ == "__main__":
    unittest.main()
