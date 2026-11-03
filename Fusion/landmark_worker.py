"""Run only with the separate MediaPipe Python interpreter."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import numpy as np
    from fusion_vsl.landmark_encoder import build_landmark_detector
    from fusion_vsl.inference_preprocessing import extract_external_landmarks
    cfg = json.loads((ROOT / "preprocessing.json").read_text(encoding="utf-8"))
    recipe = json.loads((ROOT / "landmark_recipe.json").read_text(encoding="utf-8"))
    with np.load(args.clip, allow_pickle=False) as archive:
        clip = {name: archive[name].copy() for name in ["frames", "frame_indices", "valid_mask", "padding_mask"]}
        clip["video_id"] = str(archive["video_id"].item())
        clip["quality_metadata"] = json.loads(str(archive["quality_metadata"].item()))
    with build_landmark_detector(cfg["landmark_extraction"]) as detector:
        expected = recipe["expected_provenance"]
        for name in ["mediapipe_version", "source_graph_sha256", "derived_graph_sha256", "model_sha256"]:
            if detector.provenance[name] != expected[name]:
                raise ValueError("MediaPipe recipe mismatch: " + name)
        features, detection, quality = extract_external_landmarks(clip, detector)
    np.savez_compressed(args.output, landmark_features=features, detection_mask=detection,
                        quality_json=np.asarray(json.dumps(quality, ensure_ascii=False)))


if __name__ == "__main__":
    main()
