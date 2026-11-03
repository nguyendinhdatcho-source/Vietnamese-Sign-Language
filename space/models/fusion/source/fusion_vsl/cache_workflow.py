"""M05 orchestration helpers. Producer dependencies are imported only on demand."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

from .config import sha256_file
from .feature_cache import make_recipe, require, FeatureCache, portable_path, fingerprint

REQUIRED_IDS = ("708519", "779112", "837715", "915809", "133326_1", "497887")


def select_train(context, count=12):
    require(12 <= count <= 20, "M05 smoke requires 12-20 TRAIN videos")
    train = {v:r for v,r in context.rows.items() if r["split"] == "train"}
    require(set(REQUIRED_IDS).issubset(train), "mandatory train IDs missing")
    chosen = list(REQUIRED_IDS)
    labels = {train[v]["label_id"] for v in chosen}
    groups = {train[v]["group_id"] for v in chosen}
    for vid in sorted(train):
        r = train[vid]
        if r["label_id"] in labels or r["group_id"] in groups:
            continue
        chosen.append(vid)
        labels.add(r["label_id"])
        groups.add(r["group_id"])
        if len(chosen) == count:
            return chosen
    raise ValueError("M05 insufficient distinct train classes/groups")


def producer_recipe(modality, clip, cfg, project, config_path):
    """Inspect installed models/config/code before inference, including on resume."""
    import cv2
    import numpy as np
    package = Path(__file__).parent
    files = ["feature_cache.py", "cache_workflow.py", "video.py", "sampling.py", "rgb_encoder.py"]
    settings = {"seed":cfg["seed"], "numpy_version":np.__version__, "opencv_version":cv2.__version__,
                "video_processing":cfg["video_processing"], "device":"cpu"}
    if modality == "rgb":
        from .rgb_encoder import ENCODER_VERSION, PREPROCESS_VERSION, WEIGHTS_SHA256
        weight_path = cfg["rgb_extraction"].get("weights_path")
        require(bool(weight_path), "M05 worker requires explicit verified offline weights_path")
        weight_path = (Path(config_path).parent / weight_path).resolve()
        require(sha256_file(weight_path) == WEIGHTS_SHA256, "ImageNet weights checksum mismatch")
        expected = {"encoder_version":ENCODER_VERSION, "rgb_preprocessing_version":PREPROCESS_VERSION,
                    "weights_sha256":WEIGHTS_SHA256, "weights_identifier":"imagenet", "weights_verified":True,
                    "tensorflow_version":importlib.metadata.version("tensorflow"),
                    "keras_version":importlib.metadata.version("keras")}
        settings["rgb_extraction"] = {k:v for k,v in cfg["rgb_extraction"].items() if k != "weights_path"}
    else:
        import mediapipe as mp
        from .landmark_encoder import VERSION, GRAPH_VERSION, _pose_hands_graph, validate_landmark_config
        require(mp.__version__ == "0.10.14", "M04 requires MediaPipe 0.10.14")
        root = Path(mp.__file__).parent
        options = validate_landmark_config(cfg["landmark_extraction"])
        graph, source = _pose_hands_graph(root, options["num_threads"])
        models = ("pose_detection/pose_detection.tflite", "pose_landmark/pose_landmark_full.tflite",
                  "hand_landmark/hand_landmark_full.tflite", "holistic_landmark/hand_recrop.tflite")
        expected = {"module_version":VERSION, "graph_version":GRAPH_VERSION, "mediapipe_version":mp.__version__,
                    "source_graph_sha256":source, "derived_graph_sha256":hashlib.sha256(graph.SerializeToString()).hexdigest(),
                    "model_sha256":{name:sha256_file(root/"modules"/name) for name in models},
                    "options_sha256":hashlib.sha256(json.dumps(options, sort_keys=True).encode()).hexdigest(),
                    "layout":"Pose[0:33],LeftHand[33:54],RightHand[54:75]; each [x,y,z,confidence]"}
        settings["landmark_extraction"] = options
        settings["protobuf_version"] = importlib.metadata.version("protobuf")
        files.append("landmark_encoder.py")
    hashes = {name:sha256_file(package/name) for name in files}
    hashes["worker"] = sha256_file(Path(project)/"scripts/04_cache_worker.py")
    q = clip["quality_metadata"]
    return make_recipe(modality, m02_version=q["preprocessing"]["preprocess_version"],
                       m02_sha256=q["preprocessing_sha256"], expected_provenance=expected,
                       settings=settings, code_hashes=hashes)


def open_store_reference(root, reference, context):
    """Relative path + expected recipe hash: portable across machines/mounts."""
    path = portable_path(root, reference["recipe"])
    recipe = json.loads(path.read_text(encoding="utf-8"))
    require(fingerprint(recipe) == reference["recipe_sha256"], "stale/tampered recipe reference")
    store = FeatureCache(root, recipe, context)
    require(store.recipe_path == path, "recipe reference path mismatch")
    return store


def store_reference(store):
    return {"recipe":store.recipe_path.relative_to(store.root).as_posix(),
            "recipe_sha256":fingerprint(store.recipe)}
