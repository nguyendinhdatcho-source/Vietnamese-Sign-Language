"""
Danh sách mô hình: mỗi thư mục trong models/ có model.json là một mô hình.

model.json:
  type        loại mô hình, khóa trong TYPES (vd "motion_diff")
  title       tên hiện trên web
  description dòng mô tả ngắn
  order       thứ tự trong danh sách (nhỏ đứng trước), mặc định 100
  capture     {"fps": 13, "frames": 30} – trình duyệt ghi bao nhiêu frame, tốc độ nào
  ...         các khóa riêng của loại mô hình (vd "config", "weights")
Nhãn lớp lấy từ classes.json trong thư mục mô hình, không có thì dùng models/classes.json.

Thêm loại mô hình mới: viết một lớp có
  __init__(folder, spec, num_classes)    nạp trọng số
  predict(frames) -> (probs, extra)      frames: list ảnh vuông BGR; extra: dict trả thêm cho web
  visual(spec) (staticmethod, tùy chọn)  nhãn cho phần ảnh "diffs"; không có thì web ẩn phần này
  visual_note(spec) (tùy chọn)           một câu giải thích phần ảnh đó
rồi thêm vào TYPES.
"""
import json
import os
import threading

from fusion import Fusion
from motion_diff import MotionDiff
from resnet_lstm import ResNetLSTM
from siformer import Siformer

TYPES = {
    "motion_diff": MotionDiff,
    "fusion": Fusion,
    "resnet_lstm": ResNetLSTM,
    "siformer": Siformer,
}

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_CAPTURE = {"fps": 13, "frames": 30}


def _read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class Registry:
    def __init__(self, root=MODELS_DIR):
        self.specs, self.loaded, self.lock = {}, {}, threading.Lock()
        for name in sorted(os.listdir(root)):
            folder = os.path.join(root, name)
            path = os.path.join(folder, "model.json")
            if not os.path.isfile(path):
                continue
            spec = _read_json(path)
            if spec.get("type") not in TYPES:
                print(f"Bỏ qua {name}: type {spec.get('type')!r} chưa có trong registry.TYPES")
                continue
            classes = os.path.join(folder, "classes.json")
            spec["_names"] = _read_json(classes if os.path.isfile(classes) else os.path.join(root, "classes.json"))
            spec["_folder"] = folder
            self.specs[name] = spec
        self.ids = sorted(self.specs, key=lambda k: (self.specs[k].get("order", 100), k))
        if not self.ids:
            raise SystemExit(f"Không có mô hình nào trong {root}")

    def describe(self):
        out = []
        for k in self.ids:
            spec, cls = self.specs[k], TYPES[self.specs[k]["type"]]
            out.append({
                "id": k, "title": spec.get("title", k), "description": spec.get("description", ""),
                "capture": {**DEFAULT_CAPTURE, **spec.get("capture", {})},
                "visual": cls.visual(spec) if hasattr(cls, "visual") else None,
                "visual_note": cls.visual_note(spec) if hasattr(cls, "visual_note") else "",
            })
        return out

    def get(self, k):
        """Nạp mô hình lần đầu được dùng, các lần sau dùng lại."""
        with self.lock:
            if k not in self.loaded:
                spec = self.specs[k]
                print(f"Đang nạp {k}...")
                self.loaded[k] = TYPES[spec["type"]](spec["_folder"], spec, len(spec["_names"]))
            return self.loaded[k], self.specs[k]["_names"]
