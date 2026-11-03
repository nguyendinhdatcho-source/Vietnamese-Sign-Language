"""M03: frozen ImageNet MobileNetV2 on the unchanged shared M02 clip.

Runtime features are float32. Multimodal joins and persistent dataset caches
belong to M05; this module neither samples videos nor filters low-quality clips.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np

from .config import sha256_file
from .video import PREPROCESS_VERSION as M02_VERSION

PREPROCESS_VERSION = "m03_mobilenet_v2_rgb_uint8_v1"
ENCODER_VERSION = "mobilenet_v2_alpha1_224_imagenet_avg_v1"
WEIGHTS_FILENAME = "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
WEIGHTS_URL = "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/" + WEIGHTS_FILENAME
# Measured from the official HTTPS object on 2026-10-20; also matches Keras cache.
# Evidence: artifacts/m03/weights/official_weights_verification.json.
WEIGHTS_SHA256 = "f8aff69536bd77a692c594f559c798c19bf7f3f36668fc9fa00b21c6aab4797c"
MAX_BATCH_SIZE = 16


class PretrainedWeightsError(RuntimeError):
    """Pretrained integration is BLOCKED, never replaced by random weights."""


def encoder_weight_fingerprint(encoder):
    """Digest all tensors, including frozen BatchNorm moving statistics."""
    digest = hashlib.sha256()
    for variable in encoder.weights:
        value = np.asarray(variable.numpy())
        digest.update(str(value.shape).encode())
        digest.update(value.dtype.str.encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def build_rgb_encoder(weights_path=None):
    """Frozen MobileNetV2; optional offline file must match official SHA.

    Default uses Keras weights='imagenet', verified before and after loading.
    Offline Kaggle may mount the official .h5 and pass its path.
    There is no weights=None option or fallback.
    """
    import tensorflow as tf
    if tf.keras.backend.image_data_format() != "channels_last":
        raise ValueError("M03 requires Keras channels_last (RGB NHWC)")
    if tf.keras.mixed_precision.global_policy().name != "float32":
        raise ValueError("M03 requires float32 policy; configure before building the encoder")
    try:
        if weights_path is None:
            path = Path(tf.keras.utils.get_file(
                WEIGHTS_FILENAME, WEIGHTS_URL, cache_subdir="models",
                file_hash=WEIGHTS_SHA256, hash_algorithm="sha256"))
            argument = "imagenet"
        else:
            path = Path(weights_path).expanduser().resolve()
            argument = str(path)
        if sha256_file(path) != WEIGHTS_SHA256:
            raise PretrainedWeightsError("Official ImageNet weights SHA-256 mismatch")
        model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3), alpha=1.0, include_top=False,
            pooling="avg", weights=argument)
        if sha256_file(path) != WEIGHTS_SHA256:
            raise PretrainedWeightsError("Weights changed while loading")
    except Exception as error:
        raise PretrainedWeightsError(f"M03 pretrained integration BLOCKED: {error}") from error
    model.trainable = False
    if model.input_shape != (None, 224, 224, 3) or model.output_shape != (None, 1280):
        raise ValueError("M03 unexpected MobileNetV2 model shape")
    model.rgb_provenance = {
        "encoder_version": ENCODER_VERSION, "model_name": "MobileNetV2",
        "alpha": 1.0, "include_top": False, "pooling": "avg",
        "weights_identifier": "imagenet", "weights_sha256": WEIGHTS_SHA256,
        "weights_url": WEIGHTS_URL, "weights_verified": True,
        "weights_loading": "keras_imagenet" if weights_path is None else "verified_offline_file",
        "model_weights_sha256": encoder_weight_fingerprint(model),
        "tensorflow_version": tf.__version__, "keras_version": tf.keras.__version__,
        "model_input_shape": list(model.input_shape),
        "model_output_shape": list(model.output_shape), "dtype_policy": "float32",
        "trainable": False,
    }
    return model


def preprocess_rgb_frames(frames):
    """Only M02 uint8 [16,224,224,3]; copy then Keras preprocessing ONCE."""
    import tensorflow as tf
    if not isinstance(frames, np.ndarray) or frames.shape != (16, 224, 224, 3):
        raise ValueError("M03 expected RGB frames shape (16,224,224,3)")
    if frames.dtype != np.uint8:
        raise TypeError("M03 expects uint8; float/normalized input would normalize twice")
    result = tf.keras.applications.mobilenet_v2.preprocess_input(
        frames.astype(np.float32, copy=True), data_format="channels_last")
    result = np.asarray(result, dtype=np.float32)
    if not np.isfinite(result).all() or result.min() < -1 or result.max() > 1:
        raise ValueError("M03 preprocessing must produce finite values in [-1,1]")
    return result


def _validate_clip(clip):
    if not isinstance(clip, dict) or not isinstance(clip.get("video_id"), str) or not clip["video_id"]:
        raise ValueError("M03 requires a prepared clip with nonempty string video_id")
    indices = clip.get("frame_indices")
    if not isinstance(indices, np.ndarray) or indices.shape != (16,) or indices.dtype != np.int64:
        raise ValueError("M03 frame_indices must be int64 (16,)")
    for name in ("valid_mask", "padding_mask"):
        value = clip.get(name)
        if not isinstance(value, np.ndarray) or value.shape != (16,) or value.dtype != np.bool_:
            raise ValueError(f"M03 {name} must be bool (16,)")
    valid, padding = clip["valid_mask"], clip["padding_mask"]
    n = int(valid.sum())
    if n == 0 or not np.array_equal(padding, ~valid) or not np.array_equal(valid, np.arange(16) < n):
        raise ValueError("M03 masks must be complementary, real prefix then repeat-last padding")
    if np.any(indices < 0) or np.any(np.diff(indices[:n]) <= 0) or np.any(indices[n:] != indices[n-1]):
        raise ValueError("M03 indices violate M02 temporal order/repeat-last padding")
    quality = clip.get("quality_metadata", {})
    if not isinstance(quality, dict) or not isinstance(quality.get("preprocessing"), dict):
        raise ValueError("M03 requires M02 quality/preprocessing metadata")
    if quality["preprocessing"].get("preprocess_version") != M02_VERSION:
        raise ValueError("M03 missing/unsupported M02 preprocessing version")
    digest = quality.get("preprocessing_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("M03 missing M02 preprocessing checksum")
    if quality.get("padding_count") != 16-n or quality.get("sampled_original_indices") != indices.tolist():
        raise ValueError("M03 QC metadata disagrees with indices/masks")
    if type(quality.get("valid_frames")) is not int or min(16, quality["valid_frames"]) != n:
        raise ValueError("M03 QC real frame count disagrees with masks")
    return quality


def extract_rgb_features(clip, encoder, batch_size=8, *, manifest_row,
                         min_real_frames=12, high_padding_fraction=0.25):
    """Extract ONE clip in order, retaining masks and full M02 QC metadata.

    Required manifest_row preserves split/label/group/source identity.
    Informational quality flags never reject/drop clips. Supply the SAME clip
    to the future landmark encoder. GPU work is synchronized before timing ends.
    """
    started = perf_counter()
    if type(batch_size) is not int or not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"M03 batch_size must be an integer in [1,{MAX_BATCH_SIZE}]")
    if type(min_real_frames) is not int or not 1 <= min_real_frames <= 16:
        raise ValueError("M03 min_real_frames must be an integer in [1,16]")
    if (isinstance(high_padding_fraction, bool)
            or not isinstance(high_padding_fraction, (int, float))
            or not 0 < high_padding_fraction <= 1):
        raise ValueError("M03 high_padding_fraction must be in (0,1]")
    quality = _validate_clip(clip)
    if not isinstance(manifest_row, dict) or manifest_row.get("video_id") != clip["video_id"]:
        raise ValueError("M03 manifest/clip video_id mismatch")
    for key in ("split", "label_id", "group_id", "content_hash", "relative_path"):
        if key not in manifest_row or manifest_row[key] is None or str(manifest_row[key]).strip() == "":
            raise ValueError(f"M03 missing manifest field: {key}")
    if manifest_row["split"] not in ("train", "valid", "test"):
        raise ValueError("M03 invalid split")
    label = manifest_row["label_id"]
    if isinstance(label, bool) or not str(label).isdigit() or not 0 <= int(label) < 100:
        raise ValueError("M03 invalid label_id")
    if getattr(encoder, "trainable", True) or getattr(encoder, "output_shape", None) != (None, 1280):
        raise ValueError("M03 requires a frozen encoder with 1280 output dimensions")
    provenance = getattr(encoder, "rgb_provenance", None)
    if not isinstance(provenance, dict) or not provenance.get("encoder_version"):
        raise ValueError("M03 encoder provenance is required (mock tests must identify their fixture)")
    frames = preprocess_rgb_frames(clip["frames"])
    preprocessing_seconds = perf_counter() - started
    chunks = []
    inference_started = perf_counter()
    for offset in range(0, 16, batch_size):
        output = encoder(frames[offset:offset+batch_size], training=False)
        values = np.asarray(output.numpy() if hasattr(output, "numpy") else output)
        expected = (min(batch_size, 16-offset), 1280)
        if values.shape != expected or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError(f"M03 invalid encoder output at frame {offset}: expected finite float32 {expected}")
        chunks.append(values)
    features = np.concatenate(chunks, axis=0)
    inference_seconds = perf_counter() - inference_started
    n = int(clip["valid_mask"].sum())
    flags = []
    if quality["valid_frames"] < min_real_frames:
        flags.append("few_real_frames")
    if (16-n)/16 >= high_padding_fraction:
        flags.append("high_padding")
    result = {
        "features": features, "video_id": clip["video_id"],
        "split": manifest_row["split"], "label_id": int(label),
        "source_metadata": deepcopy(manifest_row),
        "quality_metadata": deepcopy(quality),
        "quality_flags": flags,
        "quality_flag_policy": {"min_real_frames": min_real_frames,
                                "high_padding_fraction": high_padding_fraction,
                                "action": "flag_only_keep_clip"},
        "provenance": {**deepcopy(provenance),
                       "rgb_preprocessing_version": PREPROCESS_VERSION,
                       "m02_preprocessing_version": M02_VERSION,
                       "m02_preprocessing_sha256": quality["preprocessing_sha256"],
                       "features_shape": [16, 1280], "features_dtype": "float32",
                       "batch_size": batch_size, "training": False},
        "timings": {"preprocessing_validation_seconds": preprocessing_seconds,
                    "inference_seconds": inference_seconds,
                    "total_m03_seconds": perf_counter()-started},
    }
    for key in ("frame_indices", "valid_mask", "padding_mask"):
        result[key] = clip[key].copy()
    for key in ("features", "frame_indices", "valid_mask", "padding_mask"):
        result[key].setflags(write=False)
    return result


build_mobilenet_v2_encoder = build_rgb_encoder
preprocess_rgb_for_mobilenet = preprocess_rgb_frames
