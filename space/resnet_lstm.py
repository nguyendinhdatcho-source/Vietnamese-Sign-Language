"""
Loại mô hình "resnet_lstm": ResNet-18 (KerasHub, preset resnet_18_imagenet) theo từng frame + LSTM(256).

Dựng lại đúng lớp ResNet18LSTM trong notebook Polish/datcode2.ipynb (cell 4) và tiền xử lý lúc đánh giá
(read_video + transform_frames với training=False). ResNetImageConverter của KerasHub đòi tensorflow-text
(không có trên Windows) nên được viết lại: Resizing bicubic rồi x * scale + offset, không có trọng số.
"""
import base64
import json
import os

import cv2
import numpy as np

H = W = 224
# Cấu hình image_converter.json của preset resnet_18_imagenet (chuẩn hóa mean/std ImageNet)
SCALE = [0.017124753831663668, 0.01750700280112045, 0.017429193899782133]
OFFSET = [-2.1179039301310043, -2.0357142857142856, -1.8044444444444445]


def build_model(backbone_config, cfg):
    import tensorflow as tf
    import keras
    import keras_hub
    from keras import layers

    T = cfg["num_frames"]

    class Converter(layers.Layer):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.resizing = layers.Resizing(H, W, interpolation="bicubic", crop_to_aspect_ratio=True)

        def call(self, x):
            return self.resizing(x) * np.array(SCALE, "float32") + np.array(OFFSET, "float32")

    class ResNet18LSTM(keras.Model):
        def __init__(self):
            super().__init__()
            self.converter = Converter()
            self.backbone = keras_hub.models.ResNetBackbone.from_config(backbone_config)
            self.pool = layers.GlobalAveragePooling2D()
            self.temporal_dropout = layers.Dropout(cfg["temporal_dropout"])
            self.lstm = layers.LSTM(cfg["lstm_hidden"])
            self.norm = layers.LayerNormalization()
            self.dropout = layers.Dropout(cfg["classifier_dropout"])
            self.head = layers.Dense(cfg["num_classes"], dtype="float32")

        def call(self, video, training=False):
            shape = tf.shape(video)
            flat = self.converter(tf.reshape(video, (-1, H, W, 3)))
            features = self.pool(self.backbone(flat, training=training))
            sequence = self.temporal_dropout(tf.reshape(features, (shape[0], T, -1)), training=training)
            x = self.lstm(sequence, training=training)
            return self.head(self.dropout(self.norm(x), training=training))

    return ResNet18LSTM()


def is_blank(frame):
    return np.std(frame) < 5.0 and (np.mean(frame) < 15.0 or np.mean(frame) > 240.0)


def sample_frames(frames, T):
    """read_video(training=False): bỏ frame trống, BGR -> RGB, lấy đều T frame."""
    kept = [f for f in frames if not is_blank(f)] or frames
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in kept]
    return [rgb[i] for i in np.linspace(0, len(rgb) - 1, T, dtype=int)]


def center_crop(frame):
    """transform_frames(training=False): phóng cạnh ngắn về 224 rồi cắt giữa."""
    h, w = frame.shape[:2]
    s = max(H / h, W / w)
    r = cv2.resize(frame, (round(w * s), round(h * s)))
    y, x = (r.shape[0] - H) // 2, (r.shape[1] - W) // 2
    return r[y:y + H, x:x + W]


class ResNetLSTM:
    def __init__(self, folder, spec, num_classes):
        import tensorflow as tf

        self.cfg = {**spec["config"], "num_classes": num_classes}
        with open(os.path.join(folder, "backbone.json"), encoding="utf-8") as fh:
            backbone_config = json.load(fh)["config"]
        model = build_model(backbone_config, self.cfg)
        model(tf.zeros((1, self.cfg["num_frames"], H, W, 3)))
        model.load_weights(os.path.join(folder, spec.get("weights", "model.weights.h5")))
        self.infer = tf.function(lambda x: tf.nn.softmax(model(x, training=False)))

    @staticmethod
    def visual(spec):
        return f"{spec['config']['num_frames']} FRAME · RGB"

    @staticmethod
    def visual_note(spec):
        return "Các khung hình mô hình nhận vào, đã cắt giữa về 224×224."

    def predict(self, frames):
        clip = np.stack([center_crop(f) for f in sample_frames(frames, self.cfg["num_frames"])])
        probs = self.infer(clip.astype(np.float32)[None]).numpy()[0]
        thumbs = [base64.b64encode(cv2.imencode(".png", cv2.cvtColor(cv2.resize(f, (112, 112), interpolation=cv2.INTER_AREA),
                                                                     cv2.COLOR_RGB2BGR))[1]).decode() for f in clip]
        return probs, {"diffs": thumbs}
