# Phương pháp và nguồn xác minh

> Phần đối chiếu upstream bên dưới ghi lại nguồn và wrapper PyTorch lịch sử. Runtime hiện tại là `vsl_tf/` cùng các CLI TensorFlow; source lịch sử được lưu trong `backups/pre_tensorflow_only_20261020.zip`. Bảng phương pháp vẫn mô tả lựa chọn nghiên cứu, không hàm ý wrapper cũ còn hoạt động.

Tên kết quả phải dùng: **Siformer-VSL-MediaPipe-2D, bản thích nghi compatibility,
không rectification, static inference** ở cấu hình mặc định. Chỉ đổi mô tả chế độ
inference khi policy đã được hiệu chỉnh trên validation. Không gọi kết quả này là
tái lập nguyên bản đầy đủ.

Nguồn đã đọc ngày 20/10/2026:

- Paper: https://doi.org/10.1145/3664647.3681578
- Bản đọc: https://arxiv.org/html/2503.20436v1
- Source: https://github.com/mpuu00001/Siformer/tree/979a14ed15ed0f20afd77d447ad23c4f4107a2c3
- MIT license giữ trong `vendor/siformer_official/LICENSE`.

Source upstream đã đọc được lưu trong bản sao lưu có kiểm chứng; cây `vendor/` hiện chỉ giữ các tiện ích chuẩn hóa và rectification không phụ thuộc framework. Không có checkpoint/pretrained/dataset upstream nào được tải.
Các khác biệt có chủ ý của mô hình đang hoạt động nằm trong `vsl_tf/model.py`; wrapper `vsl/model.py` cũ chỉ còn trong bản sao lưu lịch sử.

## Đối chiếu source

| Thành phần | Source chính thức | Triển khai hiện tại |
|---|---|---|
| Input | `SiFormer`, `CzechSLRDataset`: trái 21, phải 21, thân 12; xy | Giữ đúng số điểm và thứ tự; không dùng 53 điểm |
| Hidden/head | 42/42/24; head 3/3/2; decoder hidden 108, 9 heads | Giữ |
| Encoder | 3 tầng, ProbSparse, FFN 2048, dropout 0.1 | Port TensorFlow trong `vsl_tf/model.py`, đối chiếu số học với source lưu trữ |
| Decoder | 2 tầng, class query, bỏ self-attention | Port TensorFlow cùng bước forward của wrapper lịch sử |
| Position | Một giá trị học được mỗi frame, lặp theo feature | Giữ dạng đó; bỏ reset global RNG |
| Sequence | PE hard-code 204; tham số seq_len chưa truyền vào PE | Mặc định 204, wrapper hỗ trợ frames qua config |
| Loss | CrossEntropy, internal classifiers không tham gia train | Giữ CE; không class weights/loss phụ tự thêm |
| Optimizer | AdamW lr 1e-4, beta 0.9/0.999, decay 1e-8 | Giữ |
| Scheduler | 100 epochs, MultiStepLR 60/80, gamma 0.1 | Giữ |
| Cân bằng | Paper dùng SMOTE | WeightedRandomSampler theo yêu cầu, không SMOTE |
| Augmentation source | Rotate/shear/arm rotation, noise | Bỏ; chỉ speed + RGB crop/jitter được yêu cầu |
| Test selection trong train.py upstream | So sánh checkpoint bằng test | Không tái sử dụng; best chỉ từ validation |

Paper mô tả ba thành phần: rectification bàn tay, feature isolation và suy luận
thích nghi. Các internal classifiers không được huấn luyện trong phương án tác
giả sử dụng. Điều này giải thích vì sao pipeline giữ các head ngẫu nhiên trong
checkpoint, nhưng không tự thêm auxiliary loss.

## Khác biệt và sửa lỗi tương thích

1. `AttentionLayer.forward` upstream trả Tensor, trong khi PyTorch
   `TransformerEncoderLayer._sa_block` lấy phần tử `[0]` của kết quả theo giao diện
   MultiheadAttention. Gọi trực tiếp attention + residual/norm/FFN trong wrapper
   tránh vô tình broadcast frame đầu, đồng thời tránh phụ thuộc private fast paths.
   Đây là sửa tương thích có thể thay đổi kết quả so với chạy source chưa sửa,
   chưa được kiểm chứng bằng thực nghiệm.
2. Bỏ `torch.manual_seed(42)` bên trong khởi tạo PE. Dùng seed thí nghiệm; không
   reset RNG mỗi clip/epoch. PE được tạo cho frames cấu hình.
3. Thay squeeze toàn bộ bằng squeeze đúng chiều class-query, giữ batch=1.
4. Không materialize attention maps dense không dùng; vẫn giữ ProbSparse.
   ProbSparse upstream lấy mẫu key ngẫu nhiên cả lúc eval. Inference seed và thứ
   tự video cố định cho tái lập trong cùng môi trường; không gọi đó là TTA.
5. Early exit chỉ encoder, per-video/per-branch. Không triển khai nhánh decoder
   adaptive. Head/projection không train nhưng được lưu/khôi phục. Static là
   kết quả đầu tiên ưu tiên. `calibrate.py --adaptive` chọn patience chỉ bằng
   validation, bao gồm phương án static 0; `evaluate.py` nhận policy đã khóa.
6. MediaPipe Holistic thay extractor của dữ liệu upstream. Chưa có bằng chứng
   tương đương chất lượng/topology định vị; không gán điểm khác nghĩa để khớp shape.

## Topology và tọa độ

Cache lưu `(T,54,2)` cùng boolean mask `(T,54)`, sequence đầy đủ trước sampling.
Thứ tự: 21 tay trái, 21 tay phải, 12 thân/mặt. MediaPipe là trái/phải giải phẫu
người ký, không đổi theo trái/phải màn hình. Không lật ngang.

Tay theo thứ tự Siformer:
`wrist,indexTip,indexDIP,indexPIP,indexMCP,middleTip,middleDIP,middlePIP,middleMCP,ringTip,ringDIP,ringPIP,ringMCP,littleTip,littleDIP,littlePIP,littleMCP,thumbTip,thumbIP,thumbMP,thumbCMC`.
Chỉ số MediaPipe tương ứng:
`0,8,7,6,5,12,11,10,9,16,15,14,13,20,19,18,17,4,3,2,1`.

Thân/mặt:
`nose,neck,rightEye,leftEye,rightEar,leftEar,rightShoulder,leftShoulder,rightElbow,leftElbow,rightWrist,leftWrist`.
Pose indices:
`0,midpoint(11,12),5,2,8,7,12,11,14,13,16,15`.
Neck là midpoint vai suy ra có chủ ý, chỉ hợp lệ khi cả hai vai có mask hợp lệ.
Đây không phải landmark neck đo trực tiếp. Không dùng face mesh hoặc z.

Cache raw x theo chiều rộng ảnh, y=1-y_mediaPipe hướng lên, đơn vị normalized
image coordinates. Đây là quy ước adapter; cần xem skeleton trước training để
xác nhận đối với video thực. Normalize dùng hàm `normalize_single_dict` body/hand
upstream, rồi trừ 0.5. Body bbox theo vai/neck/eye; hand bbox vuông có margin.
Mask false được đưa về 0 trước normalize, rồi -0.5 sau normalize.
Hàm upstream còn dùng x=0 làm sentinel và có fallback khi body geometry suy biến;
wrapper không tuyên bố đã sửa mọi giới hạn normalization upstream.

Pose visibility threshold 0.5 là lựa chọn triển khai. MediaPipe không cung cấp
visibility từng khớp tay ở đây: tay phát hiện được thì mask các tọa độ hữu hạn.
Không nội suy mất khớp, không invent confidence, không attention padding mask
mới. Missing cả clip gây lỗi có video_id. Missing từng phần giữ sentinel.

## Rectification chưa đầy đủ

Repo có `rectify_each_hand_abduction_and_addiction`,
`rectify_thumb_flexion_and_extension`, `rectify_other_flexion_and_extension`,
nhưng không có CSV motion tables mà các hàm yêu cầu.
Không suy đoán paired joints/associated joints từ shape.

Mặc định disabled. Nếu có bảng đã xác minh, điền `aa_table`, `fe_table` và bật
enabled trước cache. AA CSV cần `ranges,paired_joint_index,associated_joint_index`;
FE thêm `fingers`. Các cột danh sách dùng Python list literals.
Indices của bảng là thứ tự MediaPipe tay nguyên thủy 0..20, KHÔNG phải thứ tự
input Siformer. Wrapper đổi thứ tự trước/sau gọi code. Tay thiếu bất kỳ điểm nào
được bỏ qua rectification, nhưng vẫn giữ trong dataset. Alpha 0.5 là default của
hàm code, không phải cam kết tối ưu paper/VSL. Bảng/chiều trục/tác động ngữ nghĩa
phải được kiểm tra trước khi bật; nhánh này chưa được chạy.

## Sampling và augmentation

Không tìm thấy sampling/preprocessing script cũ, nên không thể chứng minh trùng
một quy tắc sampling đã tồn tại. Quy tắc mới được đặt tên `uniform_floor`:
`floor(linspace(0,T-1,F) * speed)`, clamp đến T-1. Speed=1 đúng quy tắc chuẩn này.
Test sử dụng frame index biết trước kiểm tra hiệu ứng, không resample toàn clip
lần thứ hai. Cùng indices áp dụng coords/mask; RGB augmentation offline giữ nguyên
toàn bộ trục thời gian để speed được vẽ mới mỗi lần sample trong train.

Crop giữ sqrt(U[.85,1]) theo cả hai chiều, vị trí chung toàn clip. Box pixel làm
tròn tối đa nửa pixel; tỷ lệ hình học giữ theo scale chung. Jitter brightness,
contrast, saturation độc lập; contrast center chung toàn clip sau brightness.
Clamp rồi resize; đưa RGB uint8 chưa normalize vào extractor. Không jitter tọa độ.
Một biến thể RGB offline mỗi train video mặc định; variants=0 là không có RGB
augmentation, cần ghi rõ khi báo cáo. Seed, tham số, parent, split, extractor,
hash video và cache được lưu. Online speed vẫn thay đổi qua epoch.

Đổi RGB/config extractor/rectification phải tạo cache mới. Không dùng landmark
gốc để đại diện cho RGB augmented. Validation/test chỉ original, speed=1.

## Resume và đánh giá

Resume tại ranh giới epoch, không hỗ trợ mid-epoch. Vị trí data/manifest/cache/
mapping/run và epochs được phép khác config; nội dung vẫn kiểm tra bằng hash.
Cần cùng device, môi trường và source để kỳ vọng tái lập. Worker
không persistent; generator DataLoader/sampler được lưu. Checkpoint gồm RNG
Python/NumPy/Torch/CUDA, optimizer, scheduler, scaler, history, best score.
Best tie-break bằng validation loss. Epoch files bất biến; best.json/last.json
là con trỏ hash. Resume sang thư mục mới giữ kết quả cũ.

Test giữ đủ 100 lớp kể cả support bằng 0. Video lỗi input là abstention, tính sai
Top-1 và false negative cho lớp thật. Confusion matrix 100x101 có cột ERROR;
nếu không có lỗi cột cuối toàn 0. Macro metrics trung bình đủ 100 lớp,
zero division=0. Không loại lỗi khỏi mẫu số. Model lỗi/OOM/NaN dừng evaluation.
Không được điều chỉnh pipeline dựa vào test rồi vẫn gọi test đó độc lập.

Latency dùng batch=1 trên một clip validation đã trích landmark, có warmup/CUDA
sync. Không bao gồm extraction/I/O, không phải latency end-to-end. Mỗi video
test chỉ có một forward trong kết quả chính. Confidence là softmax, chưa hiệu
chỉnh xác suất. SHA256 chỉ phát hiện bản sao byte, không đảm bảo không có bản
re-encode hoặc cùng người ký giữa các split.
