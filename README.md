# VSL – nhận dạng ngôn ngữ ký hiệu Việt Nam (DAT301m)

Nhận dạng 100 từ và cụm từ trong ngôn ngữ ký hiệu Việt Nam từ video. Nhánh chính là **Motion**: chỉ dùng chuyển động
giữa các frame. Mô hình **M0**: frame difference → MobileNetV2 (ImageNet) cho từng bước → trung bình theo thời gian → Dense 100.
Repo còn có các mô hình của nhóm (Fusion, Polish, Siformer), demo web dùng chung cho tất cả, và báo cáo.

## Kết quả

Nhánh Motion, cùng split train/valid/test (3077 / 385 / 404 video):

| Cấu hình | Val Macro-F1 | Test Macro-F1 (CI 95%) | Test Top-1 |
|---|---|---|---|
| m0_diff_avg | 0.821 | 0.862 (0.841–0.918) | 94.1% |
| m1_flow_avg | 0.752 | 0.782 (0.758–0.846) | 87.9% |
| m2_diff_tconv | 0.791 | 0.865 (0.836–0.916) | 92.8% |

Các mô hình khác trong demo web (số liệu trên tập valid, lấy từ `space/models/*/model.json`):

| Mô hình | Đầu vào | Val Macro-F1 |
|---|---|---|
| Siformer | 54 điểm landmark MediaPipe, ProbSparse attention | 0.861 |
| Fusion | RGB (MobileNetV2) + landmark (MediaPipe Holistic), Fusion Attention | 0.590 |
| Polish | RGB 16 frame, ResNet-18 + LSTM(256) | — |

## Cấu trúc

Nhánh Motion:

- `make_splits.py` – tạo `splits.csv` từ `data/{train,valid,test}/<lớp>/*.mp4` theo `label_mapping.pkl`
- `precompute_motion.py` – tính trước frame difference và optical flow (.npz)
- `train_motion.py` – train và đánh giá (`--mode diff|flow`, `--temporal avg|tconv|gru`)
- `vsl_motion_local.ipynb` – chạy toàn bộ pipeline và bảng so sánh
- `vsl_motion_work/` – `splits.csv` và kết quả từng run (`metrics.json`, báo cáo, xác suất dự đoán)
- `demo_m0.py` – demo Streamlit cho M0
- `demo_webcam.py` – demo M0 qua webcam (cửa sổ OpenCV)

Chung cho cả nhóm:

- `VSL_EDA_Preprocessing_TinhGon.ipynb` – EDA và tiền xử lý dữ liệu
- `space/` – demo web nhiều mô hình (M0, Fusion, Polish, Siformer), chạy trên máy (xem [`space/README.md`](space/README.md))
- `Fushion/` – gói bàn giao mô hình Fusion Attention: mã nguồn, cấu hình, metadata train (xem [`Fushion/README.md`](Fushion/README.md))
- `Polish/datcode2.ipynb` – notebook train ResNet-18 + LSTM (TensorFlow/KerasHub)
- `reports/` – báo cáo phương pháp và thực nghiệm của M0 (`.md`, `.docx`), hình minh họa, `verified_metrics.json`;
  `build_report.py` dựng lại báo cáo từ kết quả đã lưu

## Dữ liệu và trọng số

Dữ liệu (`data/`, ~2,7 GB) và file chuyển động tiền tính không có trong repo. Trọng số có trong repo:

- `vsl_motion_work/runs/m0_diff_avg/best.weights.h5` – M0
- `Fushion/fusion_best.keras`, `space/models/fusion/fusion_best.keras` – classifier của Fusion

Các file `*.h5` khác và checkpoint TensorFlow bị `.gitignore` bỏ qua. Để chạy demo web cần đặt thêm:

| File | Lấy từ đâu |
|---|---|
| `space/models/m0_diff_avg/best.weights.h5` | chép từ `vsl_motion_work/runs/m0_diff_avg/best.weights.h5` |
| `space/models/fusion/weights/mobilenet_v2_imagenet.h5` | gói bàn giao Fusion (thư mục `weights/`) |
| `space/models/polish_resnet18_lstm/model.weights.h5` | trọng số Polish đã bỏ trạng thái optimizer |
| `space/models/siformer/run/epoch_0047.ckpt-48.data-00000-of-00001` | gói `siformer_best_model.zip` |

`space/app.py` nạp sẵn M0 lúc khởi động nên ít nhất phải có file M0. Mô hình thiếu trọng số sẽ báo lỗi khi được chọn.

## Chạy

```bash
pip install tensorflow opencv-python pandas scikit-learn matplotlib streamlit
# đặt dữ liệu vào data/ (train/, valid/, test/, label_mapping.pkl), rồi chạy notebook vsl_motion_local.ipynb
streamlit run demo_m0.py   # demo M0, cần data/test để chọn video có sẵn
python demo_webcam.py      # demo webcam: SPACE ghi một lần ký, A tự ghi khi có chuyển động, Q thoát
```

`demo_webcam.py` cần bản OpenCV có GUI (`opencv-python`, không phải `-headless`).

Demo web (cài thư viện theo [`space/README.md`](space/README.md); Fusion và Siformer cần thêm môi trường MediaPipe riêng):

```bash
cp vsl_motion_work/runs/m0_diff_avg/best.weights.h5 space/models/m0_diff_avg/
python space/app.py        # mở http://localhost:7860
```

Trên web: chọn mô hình, bấm **Ghi ký hiệu** (hoặc phím Space), có 3 giây chuẩn bị rồi ghi khoảng 2,3 giây.
Nút **Lật trái phải** lật ngang cả hình hiển thị lẫn frame gửi lên mô hình; lựa chọn được nhớ trong trình duyệt.
