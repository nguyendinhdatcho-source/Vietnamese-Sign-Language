"""
Loại mô hình "siformer": Siformer (TensorFlow) trên 54 điểm landmark XY, gói bàn giao siformer_best_model.zip.

Landmark lấy đúng nguồn lúc train: clip M02 + MediaPipe Holistic 0.10.14 của gói fusion ("landmarks" trong
model.json trỏ tới thư mục đó), rồi đổi 75x4 -> 54x2 như scripts/import_landmark_cache.py, lấy mẫu và chuẩn hóa
bằng vsl.augmentation.temporal_sample và vsl.adapter.normalize. Attention ProbSparse lấy mẫu ngẫu nhiên nên bộ
sinh số được đặt lại về seed trước mỗi lần dự đoán, như evaluate.py.
"""
import json
import sys
from pathlib import Path

import numpy as np

from fusion import VISUAL_NOTE, landmark_pipeline


def to_siformer(features, detected, valid):
    """(16, 300) + detection (16, 75) -> coords (16, 54, 2), mask (16, 54); như import_landmark_cache.py."""
    from vsl.adapter import HAND_ORDER, POSE_ORDER

    features = features.reshape(-1, 75, 4)
    xy = features[:, :, :2]
    coords = np.zeros((len(xy), 54, 2), np.float32)
    mask = np.zeros((len(xy), 54), bool)
    for start, offset in ((0, 33), (21, 54)):
        indices = np.asarray(HAND_ORDER) + offset
        coords[:, start:start + 21] = xy[:, indices]
        mask[:, start:start + 21] = detected[:, indices]
    for j, pose in enumerate(POSE_ORDER):
        if pose is not None:
            coords[:, 42 + j] = xy[:, pose]
            mask[:, 42 + j] = detected[:, pose] & (features[:, pose, 3] >= 0.5)
    mask[:, 43] = mask[:, 48] & mask[:, 49]
    coords[:, 43] = (coords[:, 48] + coords[:, 49]) / 2
    coords[:, :, 1] = 1 - coords[:, :, 1]
    mask &= valid[:, None] & np.isfinite(coords).all(axis=-1)
    coords[~mask] = 0
    if not mask.any():
        raise ValueError("Không thấy người hoặc tay trong khung hình")
    return coords, mask


class Siformer:
    def __init__(self, folder, spec, num_classes):
        import tensorflow as tf

        self.folder = Path(folder)
        self.pipeline = landmark_pipeline(self.folder.parent / spec["landmarks"])
        sys.path.insert(0, str(self.folder / "project"))
        from vsl_tf.model import SiformerVSL

        run = self.folder / "run"
        self.cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        best = json.loads((run / "best.json").read_text(encoding="utf-8"))
        meta = json.loads((run / best["checkpoint"]).read_text(encoding="utf-8"))
        self.model = SiformerVSL(self.cfg)
        self.model(tf.zeros((1, self.cfg["frames"], 54, 2)), training=False)
        self.model.build_exit_heads()
        status = tf.train.Checkpoint(model=self.model, attention_rng=self.model.attention_rng).restore(
            str(run / meta["checkpoint"]))
        status.assert_existing_objects_matched()
        status.expect_partial()  # bỏ qua optimizer và biến đếm epoch trong checkpoint

    @staticmethod
    def visual(spec):
        return "16 FRAME · LANDMARK"

    @staticmethod
    def visual_note(spec):
        return VISUAL_NOTE

    def predict(self, frames):
        import tensorflow as tf
        from vsl.adapter import normalize
        from vsl.augmentation import temporal_sample

        clip = self.pipeline.make_clip(frames)
        features, detection = self.pipeline.landmarks(clip)
        coords, mask = to_siformer(features, detection, clip["valid_mask"] & ~clip["padding_mask"])
        coords, mask, _ = temporal_sample(coords, mask, self.cfg["frames"])
        x = normalize(coords, mask)
        self.model.attention_rng.reset_from_seed(self.cfg["seed"] + 17)
        logits = self.model(tf.convert_to_tensor(x[None], tf.float32), training=False)
        probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
        return probs, {"diffs": self.pipeline.thumbs(clip["frames"], features, detection)}
