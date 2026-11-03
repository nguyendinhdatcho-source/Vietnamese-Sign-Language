# Audit trước khi chuyển TensorFlow

Ngày audit: 20/10/2026. Repo root `D:\DAT301m` không có AGENTS.md. Có
`siformer-source/` và `siformer-source.zip` là bản đóng gói source trước đó;
entry point đang dùng là file ở **root**, không phải bản sao này. Source upstream
đóng băng ở `vendor/siformer_official/` (commit
`979a14ed15ed0f20afd77d447ad23c4f4107a2c3`), để đối chiếu thuật toán;
runtime PyTorch thực tế đi qua `vsl/model.py`.

Đã đọc README, `docs/METHOD.md`, `docs/KAGGLE.md`, config, notebook Kaggle,
entry points, các module `vsl/`, source attention/encoder/decoder/model vendored,
requirements và tests. Các script CLI chỉ thực thi trong `main()`.

## Dữ liệu và đường chạy hiện tại

- `data/train`, `data/valid`, `data/test` chứa 3077/385/404 MP4 RGB, 100 thư
  mục nhãn mỗi split; `configs/labels.json` cố định mapping 0–99. Không có
  landmark cache, manifests hoặc checkpoints hiện hữu ở root.
- `prepare_data.py inventory` tạo `videos.json`, `labels.json`, SHA256 audit
  theo split; `cache` crop/color jitter **RGB train** trước MediaPipe Holistic,
  lưu NPZ `coords (T,54,2)` và `mask (T,54)`, provenance/index/errors.
- `vsl/adapter.py` ánh xạ 21 trái + 21 phải + 12 thân/mặt đúng thứ tự, y hướng
  lên, normalize bằng hàm vendored rồi trừ 0.5; thiếu khớp về sentinel -0.5.
  Kinematic rectification mặc định tắt vì thiếu motion tables đã xác minh.
- `vsl/augmentation.py` có speed qua `floor(linspace(0,T-1,F)*speed)` và clamp,
  `F=204`; tọa độ/mask cùng indices. Train speed [0.9,1.1], sampler
  WeightedRandomSampler replacement theo 1/class-count. Validation/test speed=1,
  không RGB augmentation/sampler. `vsl/data.py` kiểm tra schema, hash, mapping,
  lỗi train/val gây dừng; test lỗi được ghi riêng.
- `train.py` → `vsl/data.py` → `vsl/model.py` → `vsl/engine.py`. `calibrate.py`
  chọn static hoặc patience encoder bằng validation; `evaluate.py` chạy best
  checkpoint rồi xuất metrics/report/confusion/predictions. Notebook cũ gọi
  các CLI này bằng subprocess. Không có TTA hoặc test chọn checkpoint.

## Mô hình PyTorch có hiệu lực

`SiformerVSL` khởi tạo vendored `SiFormer` với 3 encoder riêng trái/phải/body.
Feature widths 42/42/24, heads 3/3/2; decoder ghép 108 và cross attention 9
heads. Default 3 encoder layers, 2 decoder layers, FFN 2048, dropout 0.1;
class query shape `(1,1,108)`, output projection 108→100. PE trainable
`(F,1,D)` dùng cùng một giá trị cho mọi feature trong một frame; seed toàn cục
không bị reset trong wrapper. Encoder dùng ProbSparse thực sự, không phải dense
attention: chọn key ngẫu nhiên để ước lượng độ thưa, top-u query không sắp xếp,
context mặc định là mean(V), chỉ cập nhật top-u. Điểm sparsity upstream dùng
`max(sampled scores) - sum(sampled scores)/L_K` (không chia cho sample_k).
Attention projection/out là linear có bias. Attention dropout được khai báo
upstream nhưng **không áp dụng** trong ProbSparse forward.

Encoder hiệu lực được wrapper gọi tường minh: attention residual → norm1 →
linear-ReLU-dropout-linear-dropout residual → norm2, rồi final norm. Nó tránh
vấn đề giao diện return Tensor của AttentionLayer so với `nn.TransformerEncoder`.
Decoder dùng vendored `DecoderLayer`: `norm1(tgt + dropout(tgt))` trước cross
attention (không có decoder self-attention), sau đó cross-attention residual/norm2
và FFN residual/norm3. PyTorch `nn.MultiheadAttention` cross-attention có bias,
dropout attention 0.1, output projection. Các LayerNorm dùng epsilon 1e-5.
`class_query` được lặp theo batch; logits giữ chiều batch khi B=1.

Encoder internal classifier/projection ở mỗi layer được tạo nhưng freeze, không
tham gia CE training. Early exit chỉ khi inference và patience>0, từng sample/
từng branch, dựa vào argmax các classifier liên tiếp. Static patience=0.
Kiến trúc PyTorch có tham số decoder self-attention của lớp cha **không được
forward**; migration cần quyết định giữ tham số không dùng để so tổng params
hay loại chúng và giải thích chênh lệch trainable count.

## Huấn luyện/checkpoint gốc

- Config `configs/train.json`: 100 lớp; batch 24, epochs 100, seed 379,
  AdamW lr 1e-4, betas 0.9/0.999, weight decay 1e-8;
  MultiStepLR milestones 60/80 gamma 0.1; clip global grad norm 1.0;
  AMP tắt; early stopping null; validation mỗi epoch. Loss CE logits, không
  class weights/auxiliary loss. Train/val loss, accuracy, macro-F1 được ghi.
- Best theo `(validation accuracy, -validation loss)`. `epoch_NNNN.pt` đầy đủ
  model/optimizer/scheduler/scaler/RNG/history/config/labels/identity; best/last
  JSON trỏ checkpoint SHA256. Resume từ last vào run mới ở epoch boundary, giữ
  best cũ. Code identity gồm các `.py` trong `vsl/`, `vendor/` và entry point
  PyTorch. Dataset identity gồm hash manifest/cache metadata. Không có pretrained
  checkpoint được xác minh; không thể load `.pt` trực tiếp vào TensorFlow.
- Eval giữ cả 100 lớp và video lỗi trong denominator, confusion 100×101 với
  cột ERROR, macro/weighted metrics, model-only latency tách extraction.

## Baseline và môi trường

`.venv` Python 3.11.9 có PyTorch 2.14.1+cpu, NumPy 2.4.6; chưa có TensorFlow,
MediaPipe, pandas hay scikit-learn trong môi trường này. Baseline CPU trực tiếp
đã chạy với 1 encoder/1 decoder, F=8, B=1: input `(1,8,54,2)`, logits
`(1,100)` hữu hạn; trainable 1,017,552 / total 1,028,679. Đây chỉ là forward
rút gọn, **không** là kết quả huấn luyện hay chứng minh tương đương.
TensorFlow GPU không được hỗ trợ trên Windows native sau 2.10; Kaggle Linux là
đích GPU. Phiên bản Kaggle/Python/GPU thực tế chưa có quyền truy cập từ local.

## Phần buộc thay đổi

Tách module TensorFlow cho model/attention/data/training/checkpoint/evaluation;
giữ nguyên RGB, manifest/cache, MediaPipe, adapter normalization và augmentation.
Thêm CLI TensorFlow và notebook mới, giữ entry point PyTorch cho đối chiếu. Khóa
nguồn/dataset/config trong checkpoint TensorFlow, không chấp nhận checkpoint
PyTorch như weight TensorFlow. Reproduce ProbSparse, decoder no-self-attention,
head freeze/early exit, và test số học khi môi trường cho phép.

Các khác biệt có thể ảnh hưởng số học cần kiểm nghiệm: PRNG random key của
ProbSparse, initializers/default dropout Keras so với PyTorch, AdamW, AMP loss
scaling, thứ tự reduction floating point, checkpoint RNG và DataLoader sampler.
Không tuyên bố bitwise-equivalent chỉ từ cùng shape.
