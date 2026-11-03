"""
Tiến trình landmark cho mô hình fusion – chạy bằng Python có MediaPipe 0.10.14, không phải Python của app.

  python landmark_server.py <thư mục mô hình fusion>

Mỗi dòng stdin là {"clip": npz vào, "output": npz ra}, giống landmark_worker.py của gói gốc nhưng nạp
MediaPipe một lần. Trả lời một dòng trên stdout: "@@ready" khi sẵn sàng, "@@ok" hoặc "@@err <lỗi>".
"""
import json
import hashlib
import os
import sys
from pathlib import Path

if not hasattr(hashlib, "file_digest"):  # Python < 3.11 (máy Windows 3.10); code fusion_vsl gọi hàm này
    def _file_digest(stream, name):
        h = hashlib.new(name)
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
        return h
    hashlib.file_digest = _file_digest

# stdout của tiến trình chỉ dành cho giao thức; log của thư viện C++ chuyển sang stderr
proto = os.fdopen(os.dup(1), "w", encoding="utf-8")
os.dup2(2, 1)
sys.stdout = sys.stderr


def reply(msg):
    proto.write("@@" + msg.replace("\n", " ") + "\n")
    proto.flush()


def main():
    folder = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(folder / "source"))
    import numpy as np
    from fusion_vsl.inference_preprocessing import extract_external_landmarks
    from fusion_vsl.landmark_encoder import build_landmark_detector

    cfg = json.loads((folder / "preprocessing.json").read_text(encoding="utf-8"))
    recipe = json.loads((folder / "landmark_recipe.json").read_text(encoding="utf-8"))
    try:
        detector = build_landmark_detector(cfg["landmark_extraction"])
        expected = recipe["expected_provenance"]
        for name in ["mediapipe_version", "source_graph_sha256", "derived_graph_sha256", "model_sha256"]:
            if detector.provenance[name] != expected[name]:
                raise ValueError("MediaPipe recipe mismatch: " + name)
    except Exception as e:
        return reply(f"err {e}")
    reply("ready")

    with detector:
        for line in sys.stdin:
            try:
                job = json.loads(line)
                with np.load(job["clip"], allow_pickle=False) as archive:
                    clip = {k: archive[k].copy() for k in ["frames", "frame_indices", "valid_mask", "padding_mask"]}
                    clip["video_id"] = str(archive["video_id"].item())
                    clip["quality_metadata"] = json.loads(str(archive["quality_metadata"].item()))
                features, detection, quality = extract_external_landmarks(clip, detector)
                np.savez(job["output"], landmark_features=features, detection_mask=detection,
                         quality_json=np.asarray(json.dumps(quality, ensure_ascii=False)))
                reply("ok")
            except Exception as e:
                reply(f"err {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
