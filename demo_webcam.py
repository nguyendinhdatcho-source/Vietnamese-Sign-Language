"""
Demo M0 qua webcam – thực hiện ký hiệu trước camera, mô hình đoán từ.

Chạy:
  python demo_webcam.py              # webcam mặc định
  python demo_webcam.py --camera 1

Phím:
  SPACE  đếm ngược 3 giây rồi ghi một lần ký
  A      bật/tắt chế độ tự động: bắt đầu ghi khi thấy chuyển động
  Q/ESC  thoát

Để khớp với dữ liệu train (video 224x224, 30 frame, ~13 fps, người ở giữa thấy nửa thân trên):
  - cắt hình vuông ở giữa khung webcam, đứng sao cho đầu và hai tay nằm trong khung xanh
  - lấy mẫu webcam về --fps rồi ghi đúng --frames frame
  - frame difference và lấy 16 bước dùng lại đúng hàm lúc đánh giá (giống demo_m0.py)
Hình hiển thị được lật như gương cho dễ nhìn, còn mô hình nhận ảnh gốc không lật (giống dữ liệu).
"""
import argparse
import collections
import io
import json
import os
import sys
import threading
import time
import types

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(ROOT, "vsl_motion_work")
RUN = os.path.join(WORK, "runs", "m0_diff_avg")
SIZE = 224  # kích thước lúc tiền tính chuyển động
FONT_PATH = r"C:\Windows\Fonts\arial.ttf"  # cv2.putText không vẽ được dấu tiếng Việt

GREEN, RED, YELLOW, WHITE = (60, 200, 60), (60, 60, 230), (0, 210, 255), (255, 255, 255)


# ----------------------------------------------------------------------------- mô hình
def load_model():
    import tensorflow as tf
    import train_motion as tm

    with open(os.path.join(RUN, "metrics.json"), encoding="utf-8") as fh:
        args = types.SimpleNamespace(**json.load(fh)["config"])
    args.no_pretrained = True  # trọng số lấy từ best.weights.h5, không cần tải ImageNet
    with open(os.path.join(WORK, "splits_classes.json"), encoding="utf-8") as fh:
        names = json.load(fh)
    tm.NUM_CLASSES = len(names)

    model, _ = tm.build_model(args)
    model.load_weights(os.path.join(RUN, "best.weights.h5"))
    infer = tf.function(lambda x: model(x, training=False))
    infer(tf.zeros((1, args.T, args.size, args.size, 3))).numpy()  # biên dịch trước
    return infer, args, names, tm


def to_gray(square_bgr):
    return cv2.cvtColor(cv2.resize(square_bgr, (SIZE, SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)


def predict(frames, model):
    """frames: list ảnh vuông BGR (chưa lật) -> xác suất 100 lớp, giống motion_clip của demo_m0."""
    infer, args, _, tm = model
    t0 = time.perf_counter()
    gray = [to_gray(f) for f in frames]
    diff = np.stack([cv2.absdiff(b, a) for a, b in zip(gray[:-1], gray[1:])])
    buf = io.BytesIO()
    np.savez(buf, diff=diff)
    buf.seek(0)
    x = tm.load_clip(buf, "diff", args.T, args.size, False, np.random.default_rng(0))
    probs = tm.softmax(infer(x[None]).numpy())[0]
    return probs, (time.perf_counter() - t0) * 1000


# ----------------------------------------------------------------------------- vẽ
class Painter:
    def __init__(self):
        self.fonts = {s: ImageFont.truetype(FONT_PATH, s) for s in (18, 22, 30, 64)}

    def text(self, img, items):
        """items: [(xy, chữ, cỡ, màu BGR)] – vẽ một lượt bằng PIL để hiện đúng tiếng Việt."""
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for xy, s, size, color in items:
            d.text(xy, s, font=self.fonts[size], fill=color[::-1], stroke_width=2, stroke_fill=(0, 0, 0))
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def center_square(frame):
    h, w = frame.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return frame[y0:y0 + s, x0:x0 + s], (x0, y0, s)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--fps", type=float, default=13.0, help="tốc độ lấy mẫu, gần fps của dữ liệu (~9–19, trung vị ~13)")
    ap.add_argument("--frames", type=int, default=30, help="số frame mỗi lần ký (video dữ liệu luôn có 30)")
    ap.add_argument("--countdown", type=float, default=3.0)
    ap.add_argument("--motion_thr", type=float, default=6.0,
                    help="chế độ tự động: ngưỡng trung bình |frame difference| (thang 0–255) để bắt đầu ghi")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # console Windows mặc định cp1252

    print("Đang nạp mô hình...")
    model = load_model()
    names = model[2]
    painter = Painter()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được webcam {args.camera}")
    # giữ độ phân giải mặc định: đặt lại kích thước qua DSHOW làm webcam tụt xuống ~1 fps,
    # mà 640x480 cũng đủ vì ảnh chỉ cần 224x224

    win = "VSL M0 - webcam"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    state, auto = "idle", False
    t_state = time.time()
    t_sample = 0.0
    pre_roll = collections.deque(maxlen=3)  # vài frame trước khi phát hiện chuyển động (chế độ tự động)
    clip, prev_gray, last_diff, motion = [], None, None, 0.0
    result, job = None, {}  # result: (probs, ms, số frame); job: kết quả từ luồng dự đoán

    def run_predict(frames):
        job["out"] = (*predict(frames, model), len(frames))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        square, (x0, y0, s) = center_square(frame)

        # lấy mẫu về đúng fps của dữ liệu
        sampled = now - t_sample >= 1.0 / args.fps
        if sampled:
            t_sample = now
            g = to_gray(square)
            if prev_gray is not None:
                last_diff = cv2.absdiff(g, prev_gray)
                motion = 0.7 * motion + 0.3 * float(last_diff.mean())
            prev_gray = g

        # --------------------------------------------------------------- máy trạng thái
        if state == "countdown" and now - t_state >= args.countdown:
            state, clip = "recording", []
        elif state == "idle" and auto and sampled:
            pre_roll.append(square.copy())
            if motion > args.motion_thr and len(pre_roll) == pre_roll.maxlen:
                state, clip = "recording", list(pre_roll)
                pre_roll.clear()

        if state == "recording" and sampled:
            clip.append(square.copy())
            if len(clip) >= args.frames:
                state, job = "predicting", {}
                threading.Thread(target=run_predict, args=(clip,), daemon=True).start()

        if state == "predicting" and "out" in job:
            result, state, t_state = job["out"], "cooldown", now
            probs = result[0]
            top = np.argsort(-probs)[:5]
            print(" | ".join(f"{names[k]} {probs[k]:.0%}" for k in top) + f"  ({result[1]:.0f} ms)")
        elif state == "cooldown" and now - t_state >= 1.5:  # tránh tự động ghi ngay khi vừa hạ tay
            state = "idle"
            pre_roll.clear()

        # --------------------------------------------------------------- hiển thị
        view = cv2.flip(frame, 1)
        H, W = view.shape[:2]
        bx = W - x0 - s  # vị trí khung vuông sau khi lật
        color = {"recording": RED, "countdown": YELLOW}.get(state, GREEN)
        cv2.rectangle(view, (bx, y0), (bx + s - 1, y0 + s - 1), color, 3)
        if state == "recording":
            done = len(clip) / args.frames
            cv2.rectangle(view, (bx, y0 + s - 12), (bx + int(s * done), y0 + s - 1), RED, -1)

        # khung nhỏ: frame difference hiện tại (thứ mô hình nhìn thấy)
        if last_diff is not None:
            amp = np.minimum(last_diff.astype(np.float32) * 4, 255).astype(np.uint8)  # DIFF_GAIN như lúc train
            inset = cv2.cvtColor(cv2.flip(amp, 1), cv2.COLOR_GRAY2BGR)
            k = min(160, H // 4)
            view[H - k - 10:H - 10, 10:10 + k] = cv2.resize(inset, (k, k))

        items = [((10, 10), "SPACE: ghi · A: tự động " + ("BẬT" if auto else "TẮT") + " · Q: thoát", 18, WHITE)]
        if state == "countdown":
            n = int(np.ceil(args.countdown - (now - t_state)))
            items.append(((bx + s // 2 - 18, y0 + s // 2 - 40), str(n), 64, YELLOW))
        elif state == "recording":
            items.append(((bx + 10, y0 + 10), "Đang ghi...", 22, RED))
        elif state == "predicting":
            items.append(((bx + 10, y0 + 10), "Đang dự đoán...", 22, YELLOW))
        elif auto and state == "idle":
            items.append(((bx + 10, y0 + 10), f"Chờ chuyển động ({motion:.1f}/{args.motion_thr:.0f})", 22, GREEN))

        if result is not None:
            probs, ms, _ = result
            top = np.argsort(-probs)[:5]
            items.append(((10, 45), names[top[0]], 30, GREEN if probs[top[0]] >= 0.5 else YELLOW))
            for i, k in enumerate(top):
                items.append(((10, 90 + i * 28), f"{i + 1}. {names[k]}  {probs[k]:.0%}", 18, WHITE))
            items.append(((10, 90 + 5 * 28 + 4), f"{ms:.0f} ms", 18, WHITE))
            for i, k in enumerate(top):  # thanh xác suất
                y = 90 + i * 28 + 22
                cv2.rectangle(view, (10, y), (10 + int(200 * probs[k]), y + 3), GREEN, -1)
        view = painter.text(view, items)
        cv2.imshow(win, view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord(" ") and state in ("idle", "cooldown"):
            state, t_state = "countdown", now
        elif key == ord("a"):
            auto = not auto
            pre_roll.clear()
            if state == "countdown":
                state = "idle"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
