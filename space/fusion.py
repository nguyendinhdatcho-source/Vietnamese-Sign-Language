"""
Loại mô hình "fusion": Fusion Attention, ghép đặc trưng RGB (MobileNetV2) và landmark (MediaPipe Holistic).

Thư mục mô hình là nguyên gói bàn giao (load_fusion.py, source/fusion_vsl, weights, recipe...). Ở đây chỉ thay
bước đọc MP4 bằng frame webcam; QC, lấy 16 frame, resize, trích đặc trưng và model đều gọi đúng hàm gốc.
MediaPipe 0.10.14 cần protobuf<5, xung đột với TensorFlow, nên landmark chạy ở một tiến trình Python riêng
(landmark_server.py) lấy từ biến môi trường MP_PYTHON.
"""
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

if not hasattr(hashlib, "file_digest"):  # Python < 3.11 (máy Windows 3.10); code fusion_vsl gọi hàm này
    def _file_digest(stream, name):
        h = hashlib.new(name)
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
        return h
    hashlib.file_digest = _file_digest

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
# Trên máy Windows: venv tạo theo requirements-landmark.txt ở thư mục cha của space/
LOCAL_MP = HERE.parent / ".venv-mp" / "Scripts" / "python.exe"
POSE, LEFT, RIGHT = (230, 160, 60), (60, 200, 60), (60, 60, 230)  # BGR


def mediapipe_python():
    path = os.environ.get("MP_PYTHON") or (str(LOCAL_MP) if LOCAL_MP.is_file() else None)
    if not path or not Path(path).is_file():
        raise RuntimeError("Không tìm thấy Python có MediaPipe 0.10.14: đặt biến môi trường MP_PYTHON")
    return path


class LandmarkProcess:
    """Một tiến trình MediaPipe sống suốt, nhận clip npz qua stdin."""

    def __init__(self, folder):
        self.proc = subprocess.Popen(
            [mediapipe_python(), "-X", "utf8", str(HERE / "landmark_server.py"), str(folder)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")
        self._reply()  # chờ @@ready

    def _reply(self):
        for line in self.proc.stdout:
            if line.startswith("@@"):
                status, _, msg = line[2:].rstrip("\n").partition(" ")
                if status == "err":
                    raise RuntimeError("MediaPipe: " + msg)
                return
        raise RuntimeError(f"Tiến trình MediaPipe đã dừng (mã {self.proc.wait()})")

    def run(self, clip_path, out_path):
        self.proc.stdin.write(json.dumps({"clip": str(clip_path), "output": str(out_path)}) + "\n")
        self.proc.stdin.flush()
        self._reply()

    def alive(self):
        return self.proc.poll() is None


class LandmarkPipeline:
    """Clip M02 + landmark M04 của gói fusion (thư mục folder), dùng chung cho các mô hình landmark."""

    def __init__(self, folder):
        self.folder = Path(folder)
        for p in (self.folder / "source", self.folder):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        self.cfg = json.loads((self.folder / "preprocessing.json").read_text(encoding="utf-8"))
        self.process = LandmarkProcess(self.folder)

    def make_clip(self, frames):
        """Clip M02 từ frame webcam: thay decode_video của video.prepare_clip, giữ nguyên phần còn lại."""
        from fusion_vsl import video
        from fusion_vsl.inference_preprocessing import validate_prepared_clip

        rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames])
        meta = {"decoded_frames": len(rgb), "reported_frames": len(rgb), "fps": None,
                "original_frame_indices": list(range(len(rgb))), "original_shape": list(rgb[0].shape),
                "color_order": "RGB", "dtype": "uint8", "opencv_version": cv2.__version__,
                "strict_frame_count": True, "warnings": ["webcam frames"], "decode_seconds": 0.0}
        original = video.decode_video
        video.decode_video = lambda path, **kw: (rgb, meta)
        try:
            clip = video.prepare_clip("webcam", self.cfg, video_id="webcam")
        finally:
            video.decode_video = original
        return validate_prepared_clip(clip)

    def landmarks(self, clip):
        """-> landmark_features (16, 300) và detection_mask (16, 75), từ tiến trình MediaPipe."""
        with tempfile.TemporaryDirectory(prefix="landmark_") as tmp:
            src, out = Path(tmp) / "clip.npz", Path(tmp) / "landmark.npz"
            np.savez(src, **{k: clip[k] for k in ["frames", "frame_indices", "valid_mask", "padding_mask"]},
                     video_id=np.asarray(clip["video_id"]),
                     quality_metadata=np.asarray(json.dumps(clip["quality_metadata"])))
            if not self.process.alive():
                self.process = LandmarkProcess(self.folder)
            self.process.run(src, out)
            with np.load(out, allow_pickle=False) as archive:
                return archive["landmark_features"].copy(), archive["detection_mask"].copy()

    @staticmethod
    def thumbs(frames, landmarks, detection, size=112):
        """16 frame mô hình nhận vào, vẽ các điểm pose và hai tay."""
        pts = landmarks.reshape(16, 75, 4)
        out = []
        for frame, p, d in zip(frames, pts, detection):
            img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (size, size), interpolation=cv2.INTER_AREA)
            for i in np.flatnonzero(d):
                color = POSE if i < 33 else LEFT if i < 54 else RIGHT
                cv2.circle(img, (int(p[i, 0] * size), int(p[i, 1] * size)), 1, color, -1, cv2.LINE_AA)
            out.append(base64.b64encode(cv2.imencode(".png", img)[1]).decode())
        return out


_PIPELINES = {}


def landmark_pipeline(folder):
    """Một LandmarkPipeline (một tiến trình MediaPipe) cho mỗi gói fusion, dùng chung giữa các mô hình."""
    key = Path(folder).resolve()
    if key not in _PIPELINES:
        _PIPELINES[key] = LandmarkPipeline(key)
    return _PIPELINES[key]


VISUAL_NOTE = "16 khung hình mô hình nhận vào, chấm xanh dương là tư thế, xanh lá và đỏ là tay trái và tay phải."


class Fusion:
    def __init__(self, folder, spec, num_classes):
        self.folder = Path(folder)
        self.pipeline = landmark_pipeline(self.folder)
        from fusion_vsl.rgb_encoder import build_rgb_encoder
        from load_fusion import load_fusion

        self.model, _ = load_fusion(self.folder)
        self.encoder = build_rgb_encoder(self.folder / "weights" / "mobilenet_v2_imagenet.h5")

    @staticmethod
    def visual(spec):
        return "16 FRAME · LANDMARK"

    @staticmethod
    def visual_note(spec):
        return VISUAL_NOTE

    def predict(self, frames):
        from fusion_vsl.inference_preprocessing import extract_external_rgb, batch_model_inputs

        clip = self.pipeline.make_clip(frames)
        landmarks, detection = self.pipeline.landmarks(clip)
        rgb = extract_external_rgb(clip, self.encoder, batch_size=8)
        inputs = batch_model_inputs(rgb, landmarks, clip["valid_mask"], "fusion")
        probs = np.asarray(self.model(inputs, training=False))[0]
        return probs, {"diffs": self.pipeline.thumbs(clip["frames"], landmarks, detection)}
