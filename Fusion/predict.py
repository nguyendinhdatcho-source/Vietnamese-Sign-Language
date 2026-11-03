"""Predict one isolated sign from MP4 or already prepared feature arrays."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))


def validate_inputs(inputs):
    import numpy as np
    for name, shape, dtype in [("rgb_features", (1, 16, 1280), np.float32),
                               ("landmark_features", (1, 16, 300), np.float32),
                               ("valid_mask", (1, 16), np.bool_)]:
        value = inputs[name]
        if value.shape == shape[1:]:
            value = value[None]
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
        inputs[name] = value
    mask = inputs["valid_mask"][0]
    if not mask.any() or not np.array_equal(mask, np.arange(16) < mask.sum()):
        raise ValueError("valid_mask must be a nonempty True prefix, then False padding")
    return inputs


def from_video(video, mp_python):
    import numpy as np
    from fusion_vsl.config import sha256_file
    from fusion_vsl.inference_config import InferenceConfig
    from fusion_vsl.inference_preprocessing import prepare_external_clip, validate_video_path, extract_external_rgb
    from fusion_vsl.rgb_encoder import build_rgb_encoder
    cfg = json.loads((ROOT / "preprocessing.json").read_text(encoding="utf-8"))
    if Path(mp_python).resolve() == Path(sys.executable).resolve():
        raise ValueError("Use a separate MediaPipe Python environment")
    limits = InferenceConfig(model="fusion")
    identity = validate_video_path(video, limits=limits)
    clip = prepare_external_clip(video, cfg, identity, limits)
    with tempfile.TemporaryDirectory(prefix="fusion_infer_") as tmp:
        prepared, result = Path(tmp) / "clip.npz", Path(tmp) / "landmark.npz"
        np.savez_compressed(prepared, **{k: clip[k] for k in ["frames", "frame_indices", "valid_mask", "padding_mask"]},
                            video_id=np.asarray(clip["video_id"]),
                            quality_metadata=np.asarray(json.dumps(clip["quality_metadata"])))
        subprocess.run([str(mp_python), "-X", "utf8", str(ROOT / "landmark_worker.py"),
                        "--clip", str(prepared), "--output", str(result)], check=True, timeout=180)
        with np.load(result, allow_pickle=False) as archive:
            landmarks = archive["landmark_features"].copy()
            quality = json.loads(str(archive["quality_json"].item()))
    encoder = build_rgb_encoder(ROOT / "weights/mobilenet_v2_imagenet.h5")
    rgb = extract_external_rgb(clip, encoder, batch_size=8)
    if sha256_file(video) != identity["file_sha256"]:
        raise ValueError("Input video changed during preprocessing")
    return {"rgb_features": rgb, "landmark_features": landmarks, "valid_mask": clip["valid_mask"]}, quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="One MP4 containing one sign")
    source.add_argument("--features", type=Path, help="NPZ with rgb_features, landmark_features, valid_mask")
    parser.add_argument("--mediapipe-python", type=Path, help="Separate Python with requirements-landmark.txt")
    args = parser.parse_args()
    import numpy as np
    quality = None
    if args.video:
        if not args.mediapipe_python or not args.mediapipe_python.is_file():
            parser.error("--video requires --mediapipe-python pointing to an existing Python executable")
        inputs, quality = from_video(args.video.resolve(), args.mediapipe_python.resolve())
    else:
        with np.load(args.features, allow_pickle=False) as archive:
            inputs = {name: archive[name].copy() for name in ["rgb_features", "landmark_features", "valid_mask"]}
    inputs = validate_inputs(inputs)
    from load_fusion import load_fusion
    model, names = load_fusion()
    probabilities = np.asarray(model(inputs, training=False))[0]
    if probabilities.shape != (100,) or not np.isfinite(probabilities).all() or not np.isclose(probabilities.sum(), 1, atol=1e-4):
        raise ValueError("Invalid classifier probabilities")
    top = np.argsort(-probabilities, kind="stable")[:3]
    result = {"model": "fusion_attention_cw_local", "best_epoch": 14,
              "top3": [{"label_id": int(i), "label": names[i], "probability": float(probabilities[i])} for i in top],
              "real_frames": int(inputs["valid_mask"].sum())}
    if quality is not None:
        result["landmark_quality_flags"] = quality["quality_flags"]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
