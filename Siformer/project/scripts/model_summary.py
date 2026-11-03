"""Build the TensorFlow Siformer and report all active/frozen parameter counts."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import tensorflow as tf
    from vsl.common import load_config
    from vsl_tf.model import SiformerVSL
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    a = p.parse_args()
    cfg = load_config(a.config)
    model = SiformerVSL(cfg)
    model(tf.zeros((1, cfg["frames"], 54, 2)), training=False)
    model.build_exit_heads()
    model.summary()
    print("parameters_total", model.count_params())
    print("parameters_trainable", sum(int(tf.size(v)) for v in model.trainable_variables))
    print("parameters_frozen", sum(int(tf.size(v)) for v in model.non_trainable_variables if "seed_generator" not in v.path))


if __name__ == "__main__":
    main()
