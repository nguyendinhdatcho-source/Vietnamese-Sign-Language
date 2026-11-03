"""
Demo web VSL nhiều mô hình, chạy trên máy (mở ra internet bằng Cloudflare Tunnel nếu cần).

Chạy trên máy:
  python app.py                 # rồi mở http://localhost:7860
  python app.py --port 8000

Trình duyệt lấy danh sách mô hình ở /models, ghi frame theo "capture" của mô hình đang chọn rồi gửi
JPEG lên /predict. Mô hình được nạp ở lần dùng đầu tiên (mô hình đầu danh sách nạp sẵn lúc khởi động).
"""
import argparse
import base64
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from registry import Registry

ROOT = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(ROOT, "web", "index.html")
MAX_BODY = 50 * 1024 * 1024


def make_handler(reg, lock):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                with open(PAGE, "rb") as fh:
                    self._send(200, fh.read(), "text/html; charset=utf-8")
            elif self.path == "/models":
                self._json(200, {"models": reg.describe()})
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/predict":
                return self._send(404, b"not found", "text/plain")
            n = int(self.headers.get("Content-Length", 0))
            if not 0 < n <= MAX_BODY:
                return self._json(413, {"error": "dữ liệu quá lớn"})
            try:
                body = json.loads(self.rfile.read(n))
                key = body.get("model") or reg.ids[0]
                if key not in reg.specs:
                    raise ValueError(f"không có mô hình {key}")
                frames = [cv2.imdecode(np.frombuffer(base64.b64decode(j.split(",")[-1]), np.uint8), cv2.IMREAD_COLOR)
                          for j in body["frames"]]
                if len(frames) < 2 or any(f is None for f in frames):
                    raise ValueError("cần ít nhất 2 frame JPEG hợp lệ")
            except (KeyError, ValueError, TypeError, AttributeError) as e:
                return self._json(400, {"error": str(e)})

            with lock:  # mỗi lúc một lần suy luận
                model, names = reg.get(key)
                t0 = time.perf_counter()
                probs, extra = model.predict(frames)
                ms = (time.perf_counter() - t0) * 1000
            top = np.argsort(-probs)[:5]
            print(f"[{key}] " + " | ".join(f"{names[k]} {probs[k]:.0%}" for k in top) + f"  ({ms:.0f} ms)", flush=True)
            self._json(200, {
                "model": key, "top": [{"name": names[k], "prob": float(probs[k])} for k in top],
                "ms": ms, "n_frames": len(frames), **extra,
            })

        def log_message(self, fmt, *a):  # bớt log mỗi request
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # console Windows mặc định cp1252

    reg = Registry()
    print("Mô hình:", ", ".join(reg.ids))
    reg.get(reg.ids[0])  # nạp sẵn để request đầu không phải chờ
    server = ThreadingHTTPServer((args.host, args.port), make_handler(reg, threading.Lock()))
    print(f"Mở http://localhost:{args.port}  (Ctrl+C để dừng)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
