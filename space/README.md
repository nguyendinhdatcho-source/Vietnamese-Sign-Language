# VSL Studio – nhận dạng ngôn ngữ ký hiệu Việt Nam

Thực hiện một ký hiệu trước webcam, chọn mô hình và xem 5 từ được dự đoán (100 từ và cụm từ).

## Chạy trên máy

```bash
pip install -r requirements.txt
pip install --no-deps keras-hub==0.25.1
python app.py            # mở http://localhost:7860
```

Mô hình `fusion` cần thêm MediaPipe 0.10.14, phải ở một môi trường Python riêng vì xung đột protobuf với
TensorFlow. Trên Windows tạo `.venv-mp` cạnh thư mục `space/` (app tự tìm), hoặc trỏ biến `MP_PYTHON` tới nó:

```powershell
python -m venv ..\.venv-mp
..\.venv-mp\Scripts\python.exe -m pip install -r models\fusion\requirements-landmark.txt
```

## Mở link cho người khác

Giữ `python app.py` chạy, mở thêm một cửa sổ và chạy Cloudflare Quick Tunnel (không cần tài khoản):

```powershell
& "C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://localhost:7860
```

Link `https://....trycloudflare.com` in ra là link demo (có HTTPS nên webcam chạy được). Link chỉ sống khi cả
hai cửa sổ còn mở; mỗi lần chạy lại `cloudflared` sẽ ra link mới.

## Mô hình hiện có

| Thư mục | Loại | Đầu vào |
|---|---|---|
| `models/m0_diff_avg` | `motion_diff` | frame difference, MobileNetV2 + trung bình thời gian |
| `models/polish_resnet18_lstm` | `resnet_lstm` | RGB 16 frame, ResNet-18 (KerasHub) + LSTM; dựng lại từ notebook `Polish/datcode2.ipynb`, trọng số đã bỏ trạng thái optimizer |
| `models/siformer` | `siformer` | 54 điểm landmark XY, Siformer (ProbSparse attention); nguyên gói `siformer_best_model.zip`, landmark lấy từ pipeline của `models/fusion` |
| `models/fusion` | `fusion` | RGB (MobileNetV2) + landmark (MediaPipe Holistic), Fusion Attention; nguyên gói bàn giao, xem `models/fusion/README.md` |

## Thêm mô hình

Mỗi thư mục trong `models/` có `model.json` là một mô hình; web tự hiện nó trong ô chọn.

**Cùng loại đã có** (vd một mô hình khác train bằng `train_motion.py --mode diff`): tạo `models/<tên>/`,
chép `best.weights.h5` vào và viết `model.json`:

```json
{
  "type": "motion_diff",
  "title": "Tên mô hình hiện trên web",
  "description": "Mô tả ngắn hiện dưới ô chọn",
  "order": 1,
  "capture": {"fps": 13, "frames": 30},
  "weights": "best.weights.h5",
  "config": {"backbone": "mobilenetv2", "temporal": "tconv", "T": 16, "size": 224}
}
```

`config` lấy từ `metrics.json` của lần train. Nhãn lớp dùng `models/classes.json`, hoặc đặt `classes.json`
riêng trong thư mục mô hình nếu khác.

**Loại mới**: viết một lớp như `MotionDiff` (`motion_diff.py`) hoặc `Fusion` (`fusion.py`)
(`__init__(folder, spec, num_classes)`, `predict(frames) -> (probs, extra)`, `visual(spec)`/`visual_note(spec)` tùy chọn)
rồi thêm vào `TYPES` trong `registry.py`. Nếu cần thư viện mới thì thêm vào `requirements.txt`.
