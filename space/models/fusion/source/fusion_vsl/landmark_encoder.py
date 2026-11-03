"""M04: Holistic Pose + Hands, raw coordinates + confidence, separate masks.

No face inference, resampling, normalization, cache alignment or filtering.
Requires the pinned MediaPipe CPU environment; TensorFlow is not imported here.
"""
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .config import sha256_file
from .rgb_encoder import _validate_clip as _validate_shared_clip

VERSION = "m04_holistic_pose_hands_raw_xyzc_v1"
GRAPH_VERSION = "holistic_0.10.14_without_face_v1"
GROUPS = (("pose", "pose_landmarks", 0, 33),
          ("left_hand", "left_hand_landmarks", 33, 54),
          ("right_hand", "right_hand_landmarks", 54, 75))
DEFAULT_OPTIONS = {
    "min_detection_confidence": 0.5, "min_tracking_confidence": 0.5,
    "smooth_landmarks": True, "pose_visibility_threshold": 0.5,
    "low_any_hand_ratio": 0.5, "min_real_frames": 12,
    "high_padding_fraction": 0.25, "num_threads": 2,
}


class LandmarkRuntimeError(RuntimeError):
    """Execution/invalid detector output error, NOT a missing detection."""


class LandmarkEnvironmentError(RuntimeError):
    """Required MediaPipe API/models are unavailable: integration BLOCKED."""


def validate_landmark_config(options=None):
    if options is not None and (not isinstance(options, dict) or set(options)-set(DEFAULT_OPTIONS)):
        raise ValueError("M04 unknown landmark configuration")
    result = {**DEFAULT_OPTIONS, **(options or {})}
    for key in ("min_detection_confidence", "min_tracking_confidence", "pose_visibility_threshold",
                "low_any_hand_ratio", "high_padding_fraction"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int,float)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"M04 {key} must be finite in [0,1]")
    if result["high_padding_fraction"] == 0:
        raise ValueError("M04 high_padding_fraction must be >0")
    for key, maximum in (("min_real_frames",16), ("num_threads",8)):
        if type(result[key]) is not int or not 1 <= result[key] <= maximum:
            raise ValueError(f"M04 {key} must be an integer in [1,{maximum}]")
    if type(result["smooth_landmarks"]) is not bool:
        raise ValueError("M04 smooth_landmarks must be bool")
    return result


def _pose_hands_graph(mp_root, num_threads):
    """In-memory derivative of the shipped Holistic graph, removing face nodes.

    Holistic(refine_face_landmarks=False) still computes 468 face points.
    Pin the source topology and fail closed if upstream changes. No package edits.
    """
    from mediapipe.framework import calculator_pb2
    source = mp_root/"modules/holistic_landmark/holistic_landmark_cpu.binarypb"
    graph = calculator_pb2.CalculatorGraphConfig()
    graph.ParseFromString(source.read_bytes())
    expected = ["PoseLandmarkCpu", "HandLandmarksLeftAndRightCpu",
                "SplitNormalizedLandmarkListCalculator", "FaceLandmarksFromPoseCpu"]
    if [node.calculator for node in graph.node] != expected:
        raise LandmarkEnvironmentError("M04 unexpected Holistic graph topology")
    del graph.node[2:]
    outputs = [s for s in graph.output_stream if not s.startswith("FACE_LANDMARKS:")]
    sides = [s for s in graph.input_side_packet if not s.startswith("REFINE_FACE_LANDMARKS:")]
    del graph.output_stream[:]
    graph.output_stream.extend(outputs)
    del graph.input_side_packet[:]
    graph.input_side_packet.extend(sides)
    graph.num_threads = num_threads
    return graph, sha256_file(source)


class HolisticPoseHandsDetector:
    """Context-managed CPU graph; tracking reset at EVERY video boundary."""
    def __init__(self, graph, options, provenance):
        self._graph = graph
        self.options = deepcopy(options)
        self.provenance = deepcopy(provenance)
        self.closed = False

    def reset(self):
        if self.closed:
            raise LandmarkRuntimeError("M04 detector is closed")
        self._graph.reset()

    def process(self, frame):
        if self.closed:
            raise LandmarkRuntimeError("M04 detector is closed")
        return self._graph.process(input_data={"image":frame})

    def close(self):
        if not self.closed:
            try:
                self._graph.close()
            finally:
                self.closed = True

    def __enter__(self):
        if self.closed:
            raise LandmarkRuntimeError("M04 detector is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def build_landmark_detector(options=None):
    """Pinned MediaPipe Holistic CPU graph: Pose + Left/Right Hands, NO face."""
    options = validate_landmark_config(options)
    try:
        import mediapipe as mp
        from mediapipe.python.solution_base import SolutionBase
        if mp.__version__ != "0.10.14":
            raise LandmarkEnvironmentError("M04 supports MediaPipe 0.10.14; use requirements-landmark.txt")
        root = Path(mp.__file__).parent
        config, source_hash = _pose_hands_graph(root, options["num_threads"])
        model_names = ["pose_detection/pose_detection.tflite", "pose_landmark/pose_landmark_full.tflite",
                       "hand_landmark/hand_landmark_full.tflite", "holistic_landmark/hand_recrop.tflite"]
        models = {name:sha256_file(root/"modules"/name) for name in model_names}
        provenance = {
            "module_version":VERSION, "graph_version":GRAPH_VERSION,
            "mediapipe_version":mp.__version__, "api":"Holistic CPU graph / SolutionBase",
            "source_graph_sha256":source_hash,
            "derived_graph_sha256":hashlib.sha256(config.SerializeToString()).hexdigest(),
            "model_sha256":models, "face_inference":False, "segmentation":False,
            "model_complexity":1, "options":deepcopy(options),
            "coordinate_policy":"raw image-normalized xy; separate pose-z and wrist-relative hand-z",
            "pose_z_warning":"Not metric depth; legacy Holistic docs caution against interpreting pose z as depth.",
            "handedness":"Holistic anatomical left/right; never swapped by image x or mirrored",
            "confidence":"pose visibility; hand 1 if detected, 0 if missing (not calibrated probability)",
            "layout":"Pose[0:33],LeftHand[33:54],RightHand[54:75]; each [x,y,z,confidence]",
            "time_policy":"selected frames in source order, MediaPipe nominal 30fps; reset per video",
            "padding_policy":"copy source landmarks, never process padding or count it in quality rates",
            "numpy_version":np.__version__, "protobuf_version":importlib.metadata.version("protobuf"),
        }
        graph = SolutionBase(
            graph_config=config,
            side_inputs={"model_complexity":1, "smooth_landmarks":options["smooth_landmarks"],
                         "enable_segmentation":False, "smooth_segmentation":False, "use_prev_landmarks":True},
            calculator_params={
                "poselandmarkcpu__posedetectioncpu__TensorsToDetectionsCalculator.min_score_thresh":
                    options["min_detection_confidence"],
                "poselandmarkcpu__poselandmarkbyroicpu__tensorstoposelandmarksandsegmentation__ThresholdingCalculator.threshold":
                    options["min_tracking_confidence"]},
            outputs=["pose_landmarks","left_hand_landmarks","right_hand_landmarks"])
    except Exception as error:
        raise LandmarkEnvironmentError(f"M04 detector initialization BLOCKED: {error}") from error
    return HolisticPoseHandsDetector(graph, options, provenance)


def _validate_input(clip):
    _validate_shared_clip(clip)  # Existing M02/M03 shared input contract, no resampling.
    frames = clip.get("frames")
    if not isinstance(frames,np.ndarray) or frames.shape != (16,224,224,3) or frames.dtype != np.uint8:
        raise ValueError("M04 needs M02 RGB uint8 (16,224,224,3)")
    n = int(clip["valid_mask"].sum())
    if n < 16 and not np.all(frames[n:] == frames[n-1]):
        raise ValueError("M04 padding pixels do not repeat the source frame")


def _read_result(result):
    values, mask = np.zeros((75,4),np.float32), np.zeros(75,bool)
    for name, field, start, end in GROUPS:
        if not hasattr(result, field):
            raise LandmarkRuntimeError(f"M04 result missing field {field}; not a no-detection result")
        group = getattr(result, field)
        if group is None:
            continue
        if len(group.landmark) != end-start:
            raise LandmarkRuntimeError(f"M04 malformed {name} count, expected {end-start}")
        for index, point in enumerate(group.landmark, start):
            confidence = point.visibility if name == "pose" else 1.0
            coords = [point.x, point.y, point.z, confidence]
            if not np.isfinite(coords).all() or not 0 <= confidence <= 1:
                raise LandmarkRuntimeError(f"M04 invalid {name} coordinates/visibility")
            values[index] = coords
            mask[index] = True  # Returned point, even when pose visibility is low.
    if not np.isfinite(values).all():
        raise LandmarkRuntimeError("M04 landmark float32 overflow")
    return values, mask


def extract_raw_landmarks(clip, detector):
    """Process real frames once in temporal order, copy repeat-last padding."""
    _validate_input(clip)
    if not isinstance(getattr(detector,"provenance",None),dict) or not detector.provenance.get("module_version"):
        raise ValueError("M04 detector provenance is required; mocks must identify themselves")
    n = int(clip["valid_mask"].sum())
    raw, detection = np.zeros((16,75,4),np.float32), np.zeros((16,75),bool)
    frame_times = np.zeros(16,np.float64)
    started = perf_counter()
    try:
        detector.reset()  # Do not carry tracking state from another sign/video.
        reset_seconds = perf_counter()-started
        for slot in range(n):
            tick = perf_counter()
            # Own buffer protects M02 even against an implementation writing its input.
            result = detector.process(clip["frames"][slot].copy())
            frame_times[slot] = perf_counter()-tick
            raw[slot], detection[slot] = _read_result(result)
    except Exception as error:
        raise LandmarkRuntimeError(
            f"M04 video_id={clip['video_id']}: execution failed: {error}") from error
    raw[n:], detection[n:] = raw[n-1], detection[n-1]
    return {"landmarks":raw, "detection_mask":detection,
            "timings":{"mediapipe_seconds":float(frame_times.sum()), "reset_seconds":reset_seconds,
                       "per_slot_mediapipe_seconds":frame_times.tolist(), "detector_calls":n,
                       "raw_extraction_seconds":perf_counter()-started}}


def preprocess_landmarks(raw):
    """No body transform/clipping: keep position and independent z semantics."""
    values, mask = raw.get("landmarks"), raw.get("detection_mask")
    if not isinstance(values,np.ndarray) or values.shape != (16,75,4) or values.dtype != np.float32:
        raise ValueError("M04 raw landmarks must be float32 (16,75,4)")
    if not isinstance(mask,np.ndarray) or mask.shape != (16,75) or mask.dtype != np.bool_:
        raise ValueError("M04 detection_mask must be bool (16,75)")
    if not np.isfinite(values).all():
        raise ValueError("M04 NaN/Inf landmarks")
    if np.any((values[...,3] < 0) | (values[...,3] > 1)):
        raise ValueError("M04 confidence outside [0,1]")
    result = values.copy()
    result[~mask] = 0
    return result.reshape(16,300), mask.copy()


def evaluate_landmark_quality(features, detection_mask, valid_mask, frame_indices, options=None):
    """Rates use ONLY real selected frames; missing hands never exclude clips."""
    options = validate_landmark_config(options)
    if not isinstance(features,np.ndarray) or features.shape != (16,300) or features.dtype != np.float32:
        raise ValueError("M04 features must be float32 (16,300)")
    features, detection_mask = preprocess_landmarks(
        {"landmarks":features.reshape(16,75,4), "detection_mask":detection_mask})
    if (not isinstance(valid_mask,np.ndarray) or valid_mask.shape != (16,) or valid_mask.dtype != np.bool_
            or not valid_mask.any()):
        raise ValueError("M04 quality needs a nonempty real-frame mask")
    if not isinstance(frame_indices,np.ndarray) or frame_indices.shape != (16,) or frame_indices.dtype != np.int64:
        raise ValueError("M04 quality needs int64 frame_indices (16,)")
    present = {name:detection_mask[:,start:end].any(axis=1) for name,_,start,end in GROUPS}
    present["any_hand"] = present["left_hand"] | present["right_hand"]
    present["both_hands"] = present["left_hand"] & present["right_hand"]
    def positions(condition):
        return [{"slot":int(i),"source_frame_index":int(frame_indices[i])}
                for i in np.flatnonzero(valid_mask & condition)]
    n = int(valid_mask.sum())
    rates = {key:float(value[valid_mask].mean()) for key,value in present.items()}
    low = ((features.reshape(16,75,4)[:,:33,3] < options["pose_visibility_threshold"])
           & detection_mask[:,:33])
    low_points = [{"slot":int(i),"source_frame_index":int(frame_indices[i]),
                   "pose_indices":np.flatnonzero(low[i]).tolist()}
                  for i in np.flatnonzero(valid_mask & low.any(axis=1))]
    flags = []
    if n < options["min_real_frames"]: flags.append("few_real_frames")
    if (16-n)/16 >= options["high_padding_fraction"]: flags.append("high_padding")
    if rates["any_hand"] < options["low_any_hand_ratio"]: flags.append("low_any_hand_detection")
    if rates["pose"] < 1: flags.append("pose_missing_some_frames")
    if low_points: flags.append("low_pose_visibility")
    return {"real_frames":n, "padding_frames":16-n, "rates":rates,
            "no_landmarks":positions(~detection_mask.any(axis=1)),
            "no_hands":positions(~present["any_hand"]),
            "missing_groups":{key:positions(~present[key]) for key in ("pose","left_hand","right_hand")},
            "low_pose_visibility":low_points, "quality_flags":flags,
            "policy":options, "action":"report_only_keep_video",
            "hand_confidence_note":"Detection indicator only; Holistic exposes no per-point hand confidence."}


def extract_landmark_features(clip, detector, *, manifest_row):
    """M02 clip + manifest -> finite float32 (16,300), independent detection masks."""
    started = perf_counter()
    _validate_input(clip)
    if not isinstance(manifest_row,dict) or manifest_row.get("video_id") != clip["video_id"]:
        raise ValueError("M04 manifest/clip video_id mismatch")
    for key in ("split","label_id","group_id","content_hash","relative_path"):
        if manifest_row.get(key) is None or str(manifest_row[key]).strip() == "":
            raise ValueError(f"M04 missing manifest {key}")
    label = manifest_row["label_id"]
    if manifest_row["split"] not in ("train","valid","test") or isinstance(label,bool) or not str(label).isdigit() or not 0 <= int(label) < 100:
        raise ValueError("M04 invalid manifest split/label")
    raw = extract_raw_landmarks(clip,detector)
    features, mask = preprocess_landmarks(raw)
    quality = evaluate_landmark_quality(features,mask,clip["valid_mask"],clip["frame_indices"],detector.options)
    result = {
        "features":features, "detection_mask":mask, "video_id":clip["video_id"],
        "split":manifest_row["split"], "label_id":int(label), "group_id":manifest_row["group_id"],
        "source_metadata":deepcopy(manifest_row), "quality_metadata":deepcopy(clip["quality_metadata"]),
        "landmark_quality":quality, "quality_flags":quality["quality_flags"].copy(),
        "provenance":{**deepcopy(detector.provenance),
                      "m02_preprocessing_version":clip["quality_metadata"]["preprocessing"]["preprocess_version"],
                      "m02_preprocessing_sha256":clip["quality_metadata"]["preprocessing_sha256"],
                      "features_shape":[16,300], "features_dtype":"float32",
                      "detection_mask_shape":[16,75],
                      "options_sha256":hashlib.sha256(json.dumps(detector.options,sort_keys=True).encode()).hexdigest()},
        "timings":{**raw["timings"],"total_m04_seconds":perf_counter()-started}}
    for key in ("frame_indices","valid_mask","padding_mask"):
        result[key] = clip[key].copy()
    for key in ("features","detection_mask","frame_indices","valid_mask","padding_mask"):
        result[key].setflags(write=False)
    return result


create_mediapipe_extractor = build_landmark_detector
normalize_landmarks = preprocess_landmarks
calculate_detection_quality = evaluate_landmark_quality
