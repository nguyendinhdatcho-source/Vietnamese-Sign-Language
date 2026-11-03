"""M11 external adapters: same M02/M03/M04 algorithms, no invented labels."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
from time import perf_counter
import numpy as np

from .config import sha256_file
from .inference_config import InferenceConfig, require
from .rgb_encoder import _validate_clip, preprocess_rgb_frames


def validate_video_path(path, *, protected_roots=(), limits=None, known_file_hashes=()):
    limits = limits or InferenceConfig()
    path = Path(path).expanduser().resolve()
    require(path.is_file() and path.suffix.lower() == ".mp4", "need an existing single MP4 file: " + str(path))
    require(not any(path.is_relative_to(Path(p).resolve()) for p in protected_roots),
            "dataset/cache/protected project paths cannot be external inference input")
    size = path.stat().st_size
    require(0 < size <= limits.max_file_bytes, "video empty or exceeds max_file_bytes")
    try:
        digest = sha256_file(path)
    except OSError as error:
        raise ValueError("M11: input cannot be read: " + str(path)) from error
    require(digest not in known_file_hashes, "video is a byte-identical member of the locked dataset")
    return {"path": str(path), "file_sha256": digest, "bytes": size,
            "video_id": "external_" + digest[:20]}


class BoundedCapture:
    """Resource wrapper for unchanged M02 decode_video; used only in child process.

    Observes OpenCV decoded BGR frames before M02 converts them; never resamples,
    filters or modifies pixels. Count/byte limits are enforced on every read,
    even when container metadata lies. Actual RSS can exceed the array budget.
    """
    def __init__(self, capture, limits):
        import cv2
        self.capture, self.limits = capture, limits
        self.frames = self.bytes = 0
        self.digest = hashlib.sha256()
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = float(fps) if math.isfinite(fps) and fps > 0 else None
        try:
            require(self.fps is not None, "cannot bound duration: FPS unavailable")
            if math.isfinite(count) and count > 0:
                require(count <= limits.max_frames and count / self.fps <= limits.max_duration_seconds,
                        "reported duration/frame count exceeds limits")
            width, height = capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if all(math.isfinite(v) and v > 0 for v in (width, height)):
                require(width * height <= limits.max_frame_pixels, "reported resolution exceeds limit")
        except Exception:
            capture.release()
            raise

    def __getattr__(self, name):
        return getattr(self.capture, name)

    def read(self):
        import cv2
        ok, frame = self.capture.read()
        if ok:
            require(isinstance(frame, np.ndarray) and frame.ndim == 3
                    and frame.shape[-1] == 3 and frame.dtype == np.uint8 and frame.size > 0,
                    "invalid/empty decoded frame")
            self.frames += 1
            self.bytes += frame.nbytes
            require(frame.shape[0] * frame.shape[1] <= self.limits.max_frame_pixels, "decoded frame resolution limit")
            require(self.frames <= self.limits.max_frames
                    and self.frames / self.fps <= self.limits.max_duration_seconds, "decoded duration/frame limit")
            require(self.bytes <= self.limits.max_decoded_bytes, "decoded array byte limit")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.digest.update(np.asarray(rgb.shape, dtype="<i4").tobytes())
            self.digest.update(rgb.tobytes())
        return ok, frame


def prepare_external_clip(path, cfg, identity, limits, *, known_content_hashes=()):
    """Call only in isolated prepare worker, not concurrently in a TF process."""
    from . import video
    require(sha256_file(path) == identity["file_sha256"], "video changed before preprocessing")
    original_cv2 = video.cv2
    captures = []

    class BudgetedOpenCV:
        def __getattr__(self, name):
            return getattr(original_cv2, name)

        def VideoCapture(self, filename):
            cap = BoundedCapture(original_cv2.VideoCapture(filename), limits)
            captures.append(cap)
            return cap

    # This single-use subprocess owns the module; restore even on decoder errors.
    # Only the capture resource boundary changes; M02 prepare/QC/sample/resize is unchanged.
    video.cv2 = BudgetedOpenCV()
    try:
        clip = video.prepare_clip(path, cfg, video_id=identity["video_id"])
    finally:
        video.cv2 = original_cv2
    require(len(captures) == 1, "expected exactly one M02 decoder")
    content_hash = captures[0].digest.hexdigest()
    require(content_hash not in known_content_hashes, "decoded content matches locked dataset")
    require(sha256_file(path) == identity["file_sha256"], "video changed while decoding")
    clip["external_metadata"] = {**identity, "content_sha256": content_hash}
    validate_prepared_clip(clip)
    return clip


def validate_prepared_clip(clip):
    from .landmark_encoder import _validate_input
    _validate_input(clip)
    require(clip["frames"].flags.c_contiguous, "prepared frames must be contiguous RGB")
    # Recompute M02 metadata checksum instead of trusting a 64-character string.
    import json
    q = clip["quality_metadata"]
    digest = hashlib.sha256(json.dumps(q["preprocessing"], sort_keys=True).encode()).hexdigest()
    require(q["preprocessing_sha256"] == digest, "M02 preprocessing metadata checksum mismatch")
    for name in ("frames", "frame_indices", "valid_mask", "padding_mask"):
        clip[name].setflags(write=False)
    return clip


def extract_external_rgb(clip, encoder, *, batch_size=8):
    """Reuse M03 validation + preprocess ONCE + original bounded encoder loop.

    M03 extract_rgb_features requires a labeled manifest, so its numerical
    components are called directly. No split, ground truth or M05 cache record.
    """
    _validate_clip(clip)
    validate_prepared_clip(clip)
    require(type(batch_size) is int and 1 <= batch_size <= 16, "RGB batch size outside 1..16")
    require(not encoder.trainable and encoder.output_shape == (None, 1280), "need frozen M03 encoder")
    frames = preprocess_rgb_frames(clip["frames"])
    chunks = []
    for offset in range(0, 16, batch_size):
        output = encoder(frames[offset:offset + batch_size], training=False)
        chunks.append(np.asarray(output.numpy() if hasattr(output, "numpy") else output))
    features = np.concatenate(chunks)
    validate_features(features, "rgb")
    return features


def extract_external_landmarks(clip, detector):
    from .landmark_encoder import extract_raw_landmarks, preprocess_landmarks, evaluate_landmark_quality
    validate_prepared_clip(clip)
    raw = extract_raw_landmarks(clip, detector)  # reset per-video, real frames only
    features, detection = preprocess_landmarks(raw)
    quality = evaluate_landmark_quality(features, detection, clip["valid_mask"],
                                         clip["frame_indices"], detector.options)
    validate_features(features, "landmark")
    return features, detection, quality


def validate_features(values, modality):
    width = {"rgb": 1280, "landmark": 300}[modality]
    require(isinstance(values, np.ndarray) and values.shape == (16, width)
            and values.dtype == np.float32 and np.isfinite(values).all(),
            modality + " features must be finite float32 (16," + str(width) + ")")
    return values


def batch_model_inputs(rgb, landmark, valid_mask, method):
    require(method in ("rgb", "landmark", "fusion", "fusion_mean", "late_fusion"), "unknown model input adapter")
    require(isinstance(valid_mask, np.ndarray) and valid_mask.shape == (16,)
            and valid_mask.dtype == np.bool_ and valid_mask.any(), "valid_mask needs real frames")
    n = int(valid_mask.sum())
    require(np.array_equal(valid_mask, np.arange(16) < n), "valid_mask must be real prefix then padding")
    result = {"valid_mask": valid_mask[None].copy()}
    for name, value in (("rgb", rgb), ("landmark", landmark)):
        if method in (name, "fusion", "fusion_mean", "late_fusion"):
            validate_features(value, name)
            result[name + "_features"] = value[None].copy()
    return result
