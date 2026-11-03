"""
Loại mô hình "motion_diff": frame difference + CNN 2D + mô-đun thời gian (train bằng train_motion.py --mode diff).

Bản rút gọn của build_model/load_clip trong train_motion.py, bỏ phần train để image không cần pandas/sklearn.
Sửa train_motion.py ở chỗ nào liên quan thì sửa cả ở đây.
"""
import base64
import os

import cv2
import numpy as np

DIFF_GAIN = 4.0  # như train_motion.DIFF_GAIN
SIZE = 224  # kích thước lúc tiền tính chuyển động (precompute_motion.py)


def build_model(cfg, num_classes):
    from tensorflow import keras
    from tensorflow.keras import layers

    T, S = cfg["T"], cfg["size"]
    inp = keras.Input((T, S, S, 3))
    if cfg["backbone"] == "mobilenetv2":
        bb = keras.applications.MobileNetV2(include_top=False, weights=None, input_shape=(S, S, 3), pooling="avg")
        x = inp  # MobileNetV2 nhận đầu vào [-1, 1]
    else:
        bb = keras.applications.EfficientNetB0(include_top=False, weights=None, input_shape=(S, S, 3), pooling="avg")
        x = layers.Rescaling(127.5, offset=127.5)(inp)  # EfficientNet của Keras nhận [0, 255]

    f = layers.TimeDistributed(bb, name="frame_encoder")(x)
    f = layers.Dropout(0.3)(f)
    if cfg["temporal"] == "avg":
        h = layers.GlobalAveragePooling1D()(f)
    elif cfg["temporal"] == "gru":
        h = layers.Bidirectional(layers.GRU(256))(f)
    else:  # tconv
        h = layers.Conv1D(256, 3, padding="same", activation="relu")(f)
        h = layers.Dropout(0.3)(h)
        h = layers.Conv1D(256, 3, padding="same", activation="relu")(h)
        h = layers.GlobalAveragePooling1D()(h)
    h = layers.Dropout(0.4)(h)
    out = layers.Dense(num_classes, dtype="float32", name="logits")(h)
    return keras.Model(inp, out)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


class MotionDiff:
    def __init__(self, folder, spec, num_classes):
        import tensorflow as tf

        self.cfg = spec["config"]
        model = build_model(self.cfg, num_classes)
        model.load_weights(os.path.join(folder, spec.get("weights", "best.weights.h5")))
        T, S = self.cfg["T"], self.cfg["size"]
        self.infer = tf.function(lambda x: model(x, training=False))
        self.infer(tf.zeros((1, T, S, S, 3))).numpy()  # biên dịch trước

    @staticmethod
    def visual(spec):
        return f"{spec['config']['T']} BƯỚC · FRAME DIFFERENCE"

    @staticmethod
    def visual_note(spec):
        return "Những vùng sáng thể hiện thay đổi giữa các khung hình."

    def steps(self, frames):
        """frames BGR vuông -> T bước frame difference uint8, giống load_clip lúc đánh giá."""
        gray = [cv2.cvtColor(cv2.resize(f, (SIZE, SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                for f in frames]
        diff = np.stack([cv2.absdiff(b, a) for a, b in zip(gray[:-1], gray[1:])])
        idx = np.clip(np.round(np.linspace(0, len(diff) - 1, self.cfg["T"])), 0, len(diff) - 1).astype(int)
        return diff[idx]

    def predict(self, frames):
        steps = self.steps(frames)
        amp = np.minimum(steps.astype(np.float32) * DIFF_GAIN, 255.0)
        S = self.cfg["size"]
        x = amp / 255.0
        if x.shape[1:] != (S, S):
            x = np.stack([cv2.resize(f, (S, S), interpolation=cv2.INTER_LINEAR) for f in x])
        x = np.repeat(x[..., None] * 2.0 - 1.0, 3, axis=-1).astype(np.float32)
        probs = softmax(self.infer(x[None]).numpy()[0])

        thumbs = [base64.b64encode(cv2.imencode(".png", cv2.resize(a.astype(np.uint8), (112, 112)))[1]).decode()
                  for a in amp]
        return probs, {"diffs": thumbs}
