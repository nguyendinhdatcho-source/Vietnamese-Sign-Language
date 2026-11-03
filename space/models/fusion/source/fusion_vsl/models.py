"""M07 untrained baselines and lazy M08 feature fusion entry points.

No encoder, cache writer, optimizer or training is invoked by this module.
Landmark-only deliberately uses raw cached xyzc and the temporal mask; its
detection mask is trace metadata, not a temporal mask or an extra model input.
"""
from dataclasses import dataclass
import math
from numbers import Real

import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class BaselineModelConfig:
    """Classifier settings, independent of the frozen M00 feature recipe."""
    projection_units: int = 128
    gru_units: int = 64
    attention_units: int = 64
    classifier_units: int = 128
    dropout: float = 0.4

    def __post_init__(self):
        for key in ("projection_units", "gru_units", "attention_units", "classifier_units"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"M07: {key} must be a positive integer")
        if (isinstance(self.dropout, bool) or not isinstance(self.dropout, Real)
                or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1):
            raise ValueError("M07: dropout must be finite in [0,1)")


def _temporal_checks(sequence, valid_mask, frames=None, feature_dim=None):
    if sequence.dtype != tf.float32 or valid_mask.dtype != tf.bool:
        raise ValueError("M07: features must be float32 and valid_mask must be bool")
    tf.debugging.assert_rank(sequence, 3, message="M07: expected (B,T,D) features")
    tf.debugging.assert_rank(valid_mask, 2, message="M07: expected (B,T) valid_mask")
    tf.debugging.assert_equal(tf.shape(sequence)[:2], tf.shape(valid_mask),
                              message="M07: features/mask batch or temporal shape mismatch")
    tf.debugging.assert_positive(tf.shape(sequence)[0], message="M07: empty batch")
    tf.debugging.assert_all_finite(sequence, "M07: NaN/Inf features")
    tf.debugging.assert_equal(tf.reduce_all(tf.reduce_any(valid_mask, axis=1)), True,
                              message="M07: all-invalid clip is not allowed")
    if frames is not None:
        tf.debugging.assert_equal(tf.shape(sequence)[1], frames, message="M07: wrong frame count")
    if feature_dim is not None:
        tf.debugging.assert_equal(tf.shape(sequence)[2], feature_dim, message="M07: wrong feature dimension")


@tf.keras.utils.register_keras_serializable(package="fusion_vsl")
class ValidateTemporalInput(tf.keras.layers.Layer):
    """Runtime contract, explicit-mask padding sanitization; never infer from zero."""
    def __init__(self, frames=16, feature_dim=1280, **kwargs):
        super().__init__(**kwargs)
        self.frames, self.feature_dim = frames, feature_dim

    def call(self, inputs):
        sequence, valid_mask = inputs
        _temporal_checks(sequence, valid_mask, self.frames, self.feature_dim)
        # M02/M05 guarantee a real prefix followed by repeat-last padding.
        prefix = tf.sequence_mask(tf.reduce_sum(tf.cast(valid_mask, tf.int32), axis=1), self.frames)
        tf.debugging.assert_equal(valid_mask, prefix, message="M07: valid_mask must be a real prefix")
        return tf.where(valid_mask[..., None], sequence, tf.zeros_like(sequence))

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        return {**super().get_config(), "frames": self.frames, "feature_dim": self.feature_dim}


@tf.keras.utils.register_keras_serializable(package="fusion_vsl")
class MaskedTemporalAttention(tf.keras.layers.Layer):
    """Additive attention: [sequence, explicit bool mask] -> (pool, weights).

    e_t = v^T tanh(W h_t + b). Invalid timesteps have exactly zero weight;
    each nonempty clip's real weights sum to one. Returning weights as a second
    tensor allows a debug Functional model to share the classifier's weights.
    """
    def __init__(self, attention_units=64, **kwargs):
        super().__init__(**kwargs)
        if type(attention_units) is not int or attention_units < 1:
            raise ValueError("M07: attention_units must be a positive integer")
        self.attention_units = attention_units
        self.supports_masking = True

    def build(self, input_shape):
        sequence_shape, mask_shape = input_shape
        if len(sequence_shape) != 3 or len(mask_shape) != 2 or sequence_shape[-1] is None:
            raise ValueError("M07: attention expects [(B,T,H), (B,T)] with known H")
        self.kernel = self.add_weight(name="kernel", shape=(sequence_shape[-1], self.attention_units),
                                      initializer="glorot_uniform")
        self.bias = self.add_weight(name="bias", shape=(self.attention_units,), initializer="zeros")
        self.context_vector = self.add_weight(name="context_vector", shape=(self.attention_units, 1),
                                              initializer="glorot_uniform")
        super().build(input_shape)

    def call(self, inputs, mask=None):
        sequence, valid_mask = inputs  # Keras implicit mask is never used as a fallback.
        _temporal_checks(sequence, valid_mask)
        clean = tf.where(valid_mask[..., None], sequence, tf.zeros_like(sequence))
        hidden = tf.tanh(tf.matmul(clean, self.kernel) + self.bias)
        scores = tf.squeeze(tf.matmul(hidden, self.context_vector), axis=-1)
        scores = tf.where(valid_mask, scores, tf.constant(-1e9, tf.float32))
        weights = tf.where(valid_mask, tf.nn.softmax(scores, axis=1), tf.zeros_like(scores))
        weights = weights / tf.reduce_sum(weights, axis=1, keepdims=True)
        pooled = tf.reduce_sum(clean * weights[..., None], axis=1)
        return pooled, weights

    def compute_output_shape(self, input_shape):
        shape = input_shape[0]
        return (shape[0], shape[2]), (shape[0], shape[1])

    def compute_mask(self, inputs, mask=None):
        return None, None

    def get_config(self):
        return {**super().get_config(), "attention_units": self.attention_units}


def _build_baseline(modality, input_shape, num_classes, config):
    dim = {"rgb": 1280, "landmark": 300}[modality]
    if tuple(input_shape) != (16, dim) or type(num_classes) is not int or num_classes != 100:
        raise ValueError(f"M07: {modality} contract requires (16,{dim}) and 100 classes")
    options = config if isinstance(config, BaselineModelConfig) else BaselineModelConfig(**(config or {}))
    name = modality + "_features"
    features = tf.keras.Input(input_shape, dtype="float32", name=name)
    valid = tf.keras.Input((16,), dtype="bool", name="valid_mask")
    clean = ValidateTemporalInput(feature_dim=dim, name="validate_temporal_input")([features, valid])
    sequence = tf.keras.layers.Dense(options.projection_units, activation="relu", name="frame_projection")(clean)
    sequence = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(options.gru_units, return_sequences=True), name="temporal_bigru")(
            sequence, mask=valid)
    pooled, _ = MaskedTemporalAttention(options.attention_units, name="temporal_attention")([sequence, valid])
    hidden = tf.keras.layers.Dense(options.classifier_units, activation="relu", name="classifier_dense")(pooled)
    hidden = tf.keras.layers.Dropout(options.dropout, name="classifier_dropout")(hidden)
    probabilities = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32",
                                          name="probabilities")(hidden)
    return tf.keras.Model({name: features, "valid_mask": valid}, probabilities, name=modality + "_baseline")


def build_rgb_model(input_shape=(16,1280), num_classes=100, *, config=None):
    """Uncompiled Functional RGB baseline; inputs rgb_features + valid_mask."""
    return _build_baseline("rgb", input_shape, num_classes, config)


def build_landmark_model(input_shape=(16,300), num_classes=100, *, config=None):
    """Uncompiled Functional landmark baseline; zero/missing real frames retained.

    Detection mask is NOT applied again: M04/M05 already zero-fill absent points
    while retaining xyzc layout. Only valid_mask controls temporal exclusion.
    """
    return _build_baseline("landmark", input_shape, num_classes, config)


def baseline_inputs(inputs, modality):
    """Select/validate two named baseline inputs from an unchanged M06 input dict.

    Extra detection/padding masks remain available to callers for trace/debug,
    but are not silently passed to a model that has only two inputs.
    """
    if modality not in ("rgb", "landmark"):
        raise ValueError("M07: modality must be rgb/landmark")
    key = modality + "_features"
    if not isinstance(inputs, dict) or key not in inputs or "valid_mask" not in inputs:
        raise ValueError("M07: missing baseline features or valid_mask")
    values, mask = tf.convert_to_tensor(inputs[key]), tf.convert_to_tensor(inputs["valid_mask"])
    _temporal_checks(values, mask, 16, 1280 if modality == "rgb" else 300)
    return {key: values, "valid_mask": mask}


def attention_debug_model(model):
    """Share a baseline's variables; output probabilities and (B,16) attention."""
    _, weights = model.get_layer("temporal_attention").output
    return tf.keras.Model(model.input, {"probabilities": model.output, "attention_weights": weights},
                          name=model.name + "_debug")


def _probabilities(values, name):
    values = np.asarray(values)
    if (values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != 100
            or not np.issubdtype(values.dtype, np.floating)):
        raise ValueError(f"M07: {name} must be floating (B,100), nonempty")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError(f"M07: {name} must contain finite probabilities in [0,1]")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError(f"M07: {name} rows must sum to one")
    return values.astype(np.float32)


def late_fusion(rgb_probs, landmark_probs, alpha=0.5, *, rgb_video_ids=None,
                landmark_video_ids=None, rgb_label_mapping_sha256=None,
                landmark_label_mapping_sha256=None):
    """Eager NumPy probability average, not M08 learned feature fusion.

    Optional identity/mapping bindings are checked when supplied. Without them
    this is numeric validation only; arrays cannot prove their ID/class order.
    Production callers should provide BOTH ID lists and BOTH mapping hashes.
    No labels, metrics or alpha search are used here.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("M07: alpha must be finite in [0,1]")
    rgb, landmark = _probabilities(rgb_probs, "rgb_probs"), _probabilities(landmark_probs, "landmark_probs")
    if rgb.shape != landmark.shape:
        raise ValueError("M07: probability shapes differ")
    if rgb_video_ids is not None or landmark_video_ids is not None:
        if rgb_video_ids is None or landmark_video_ids is None:
            raise ValueError("M07: both video ID lists are required together")
        a, b = tuple(rgb_video_ids), tuple(landmark_video_ids)
        if (a != b or len(a) != len(rgb) or any(not isinstance(v, str) or not v for v in a)
                or len(set(a)) != len(a)):
            raise ValueError("M07: late fusion video IDs/order mismatch or duplicate IDs")
    if rgb_label_mapping_sha256 is not None or landmark_label_mapping_sha256 is not None:
        a, b = rgb_label_mapping_sha256, landmark_label_mapping_sha256
        if (not isinstance(a, str) or len(a) != 64 or any(c not in "0123456789abcdef" for c in a) or a != b):
            raise ValueError("M07: late fusion label mapping mismatch/invalid hash")
    return (float(alpha) * rgb + (1.0 - float(alpha)) * landmark).astype(np.float32)


def temporal_attention_pooling(sequence, mask=None):
    """M08 convenience wrapper around the unchanged M07 attention layer."""
    if mask is None:
        raise ValueError("M08: explicit valid_mask is required")
    pooled, _ = MaskedTemporalAttention()([sequence, mask])
    return pooled

def build_feature_fusion(rgb_shape=(16,1280), landmark_shape=(16,300), num_classes=100, *, config=None):
    """M08 dual independent branches with masked attention, uncompiled."""
    from .fusion_models import build_feature_fusion as build
    return build(rgb_shape, landmark_shape, num_classes, config=config)

def build_fusion_without_attention(rgb_shape=(16,1280), landmark_shape=(16,300), num_classes=100, *, config=None):
    """M08 ablation: same branches/head, masked temporal mean instead of attention."""
    from .fusion_models import build_fusion_without_attention as build
    return build(rgb_shape, landmark_shape, num_classes, config=config)

def export_model_spec(model, output_path=None, *, label_mapping, expected_label_mapping_sha256=None,
                      manifest_sha256=None, feature_recipes=None):
    """M08 architecture/tensor/label-binding spec; does not certify training."""
    from .fusion_models import export_model_spec as export
    return export(model, output_path, label_mapping=label_mapping,
                  expected_label_mapping_sha256=expected_label_mapping_sha256,
                  manifest_sha256=manifest_sha256, feature_recipes=feature_recipes)
