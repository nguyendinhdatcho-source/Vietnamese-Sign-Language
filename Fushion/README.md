# Fusion Attention — bản có class weight train trên máy

Đây là bản **local CPU**, train ngày **23/10/2026 (giờ Việt Nam/Thái Lan)**,
run `20261022T180151Z_fusion_7841f1fe95`, dùng `class_weight_mode=balanced`.
Checkpoint tốt nhất là **epoch 14**, chọn theo **VALID val_loss nhỏ nhất**.
File trọng số được sao chép nguyên byte, không train lại.

## Gửi những gì?

**Gửi nguyên thư mục `model` này** (có thể nén thành ZIP). Các thành phần:

- `fusion_best.keras`: mô hình Fusion Attention đã train, checkpoint epoch 14.
- `label_mapping.json`: ánh xạ tên tiếng Việt → ID 0–99, đúng thứ tự đầu ra lúc train.
- `source/fusion_vsl/`: kiến trúc, custom Keras layers và pipeline tiền xử lý gốc.
- `weights/mobilenet_v2_imagenet.h5`: CNN MobileNetV2 trích đặc trưng RGB, không cần tải weights khi dự đoán.
- `preprocessing.json`, `rgb_recipe.json`, `landmark_recipe.json`: cấu hình lấy frame và trích đặc trưng.
- `load_fusion.py`: nạp classifier bằng `compile=False`, không khôi phục optimizer.
- `predict.py`, `landmark_worker.py`: chạy suy luận từ MP4 hoặc feature NPZ.
- `requirements.txt`, `requirements-landmark.txt`: thư viện cho hai môi trường riêng.
- `model_info.json`, `training_metadata/`: nguồn gốc checkpoint và metadata train để đối chiếu.
- `checksums.json`, `verification.json`: checksum và kết quả kiểm tra gói bàn giao.

Người nhận không cần dữ liệu train/valid/test, các checkpoint epoch khác hoặc môi trường `.venv` trên máy gốc.

## Lưu ý về đầu vào

`fusion_best.keras` là classifier **nhận feature**, không nhận MP4 hoặc ảnh trực tiếp:

```text
rgb_features       float32 (B, 16, 1280)
landmark_features  float32 (B, 16, 300)
valid_mask         bool    (B, 16)
output                    (B, 100)
```

`valid_mask=True` cho frame thật, `False` cho padding lặp frame cuối.
Hai nhánh phải dùng cùng 16 frame. Landmark theo thứ tự Pose 33 điểm,
tay trái 21, tay phải 21; mỗi điểm `[x,y,z,confidence]`, giữ tọa độ gốc như lúc train.
RGB dùng MobileNetV2 `preprocess_input` đưa pixel về [-1,1].
Không áp dụng chuẩn hóa ImageNet mean/std của notebook PyTorch cũ.

## Cài trên máy khác (Windows, Python 3.12)

Mở PowerShell trong thư mục `model`, tạo hai môi trường:

```powershell
py -3.12 -m venv .venv-tf
.\.venv-tf\Scripts\python.exe -m pip install -r requirements.txt
py -3.12 -m venv .venv-mp
.\.venv-mp\Scripts\python.exe -m pip install -r requirements-landmark.txt
```

TensorFlow 2.20 và MediaPipe 0.10.14 cần hai môi trường vì yêu cầu protobuf khác nhau.
MediaPipe cài kèm graph/model cần thiết; không cần chép `.venv-m04` từ máy gốc.
Các lệnh cài đặt cần Internet. Khi đã cài đủ thư viện, suy luận dùng assets local.

## Dự đoán một video MP4

```powershell
.\.venv-tf\Scripts\python.exe -X utf8 predict.py --video "D:\video\ky_hieu.mp4" --mediapipe-python ".\.venv-mp\Scripts\python.exe"
```

Chương trình in Top 3 nhãn và xác suất. Đây là nhận dạng **một ký hiệu trong một clip**;
không phải dịch câu hoặc nhận dạng liên tục từ webcam.
Mặc định chạy CPU; giới hạn clip 20 giây, 600 frame, 64 MiB, tối đa 1920×1080,
256 MiB tổng mảng frame giải mã. Clip ngắn, rõ người và hai tay sẽ phù hợp hơn.
File tạm được dọn sau xử lý; video gốc không bị sửa.

## Đã có feature đúng pipeline

Tạo NPZ chứa ba mảng `rgb_features`, `landmark_features`, `valid_mask`
(một clip, có hoặc không có chiều batch=1), rồi chạy:

```powershell
.\.venv-tf\Scripts\python.exe -X utf8 predict.py --features "D:\features\clip.npz"
```

Nếu tích hợp vào Python, chỉ cần môi trường TensorFlow cho phần này:

```python
from load_fusion import load_fusion
model, class_names = load_fusion()
probabilities = model({
    "rgb_features": rgb_features,             # (B,16,1280), float32
    "landmark_features": landmark_features,   # (B,16,300), float32
    "valid_mask": valid_mask,                 # (B,16), bool
}, training=False).numpy()
labels = [class_names[i] for i in probabilities.argmax(axis=1)]
```

Nạp qua `load_fusion()` để đăng ký custom layers và kiểm checksum/mapping.
Class weight đã được dùng trong lúc train; không cần nhân class weight vào xác suất suy luận.
Các metadata có thể chứa đường dẫn máy gốc để truy vết; runner dùng đường dẫn tương đối theo gói.

## Phạm vi kiểm tra

Xem `verification.json` để biết kiểm tra thực tế đã chạy.
Phiên bản decoder lúc trích feature lịch sử được lưu trong recipe (RGB OpenCV 4.13,
landmark OpenCV 4.11). Máy/codec khác có thể giải mã pixel khác; không mặc định coi
mọi môi trường mới sẽ cho xác suất trùng tuyệt đối.
