Bạn là một nghiên cứu sinh ngành Thị giác máy tính, viết báo cáo khoa học bằng tiếng Việt rất tốt. Hãy viết cho tôi **hai phần** của báo cáo môn học (DAT301m – Nhận dạng ngôn ngữ ký hiệu Việt Nam):

1. **MÔ TẢ PHƯƠNG PHÁP** – khoảng **3 trang A4** (Times New Roman 12, giãn dòng 1.15, tức khoảng 1.300–1.500 chữ, chưa tính hình và bảng).
2. **KẾT QUẢ THỰC NGHIỆM** – khoảng 3–4 trang.

Toàn bộ thông tin kỹ thuật và số liệu ở bên dưới. Đọc kỹ các yêu cầu trước khi viết.

---

## A. YÊU CẦU BẮT BUỘC

- **Không** dùng các chữ "M0", "baseline", "mô hình cơ sở", "phiên bản 1/2". Gọi phương pháp bằng tên đầy đủ: **Frame Difference + MobileNetV2 + Temporal Average Pooling** (có thể viết tắt sau lần đầu, ví dụ "FD-MobileNetV2").
- **Không** nhắc tới optical flow, Conv1D theo thời gian, GRU hay bất kỳ biến thể/thử nghiệm nào khác. Báo cáo chỉ nói về **một** phương pháp duy nhất này. Ở phần hướng phát triển có thể nói chung chung, ví dụ "mô-đun thời gian có khả năng mô hình hóa thứ tự".
- **Không** viết về phần demo (webcam, web).
- **Không bịa số liệu.** Chỉ dùng các con số có ở mục C–E. Nếu muốn so sánh với công trình khác, chỉ nêu ý tưởng (TSN, two-stream, MobileNetV2) và không kèm con số.
- Phần phương pháp phải **tự đứng được**: người đọc chưa xem code vẫn hiểu và có thể tái hiện. Không viết "như đã trình bày ở phần chung".

## B. VĂN PHONG MONG MUỐN

- Văn phong học thuật nhưng dễ đọc và có mạch lập luận. Mỗi lựa chọn thiết kế cần trả lời **"tại sao"** chứ không chỉ liệt kê "làm gì". Ví dụ: vì sao dùng chuyển động thay cho RGB, vì sao loại khung trống, vì sao khuếch đại ×4, vì sao đóng băng BatchNorm, vì sao chọn checkpoint theo Macro-F1.
- Mở đầu phần phương pháp bằng **động lực** (ý nghĩa của ký hiệu nằm ở chuyển động, còn phông nền và ngoại hình người ký là nhiễu), sau đó mới đi vào chi tiết.
- Phần kết quả cần **phân tích và diễn giải**, không chỉ đọc lại bảng. Hãy liên hệ ngược về các lựa chọn thiết kế, ví dụ: giai đoạn đóng băng backbone cho F1 thấp, F1 tăng vọt khi fine-tune, nghĩa là đặc trưng ImageNet phải thích nghi với ảnh sai khác.
- Công thức viết dạng LaTeX ($...$). Bảng viết bằng Markdown. Đánh dấu chỗ đặt hình là `[Hình X: chú thích]`.
- Trình bày trung thực về giới hạn, nhưng không tự hạ thấp kết quả.

## C. MÔ TẢ PHƯƠNG PHÁP (thông tin kỹ thuật)

**Bài toán:** Nhận dạng ký hiệu rời rạc (isolated sign recognition). Mỗi video chứa đúng 1 ký hiệu và cần phân vào 1 trong **100 lớp** từ/cụm từ tiếng Việt (ví dụ: Tôi, Xe máy, Bệnh viện, Cách ly, Sốt, Hôm nay…).

**Dữ liệu video:** khoảng 30 khung/video (trung vị 30, ít nhất 16), tốc độ khoảng 12 fps.

**Bước 1 – Tiền xử lý (tính một lần, offline):**
- Đọc toàn bộ khung và resize về 224×224 bằng nội suy INTER_AREA.
- Loại **khung trống** (đồng màu, thường là khung đen ở đầu hoặc cuối video): một khung bị loại nếu độ lệch chuẩn của cả 3 kênh màu đều ≤ 1 (thang 0–255). Lý do: cặp "khung đen → khung có người" tạo ra ảnh sai khác cực lớn nhưng không phải chuyển động ký hiệu. Thống kê: 754/3.866 video có khung trống, tổng cộng 9.245 khung bị loại.
- Chuyển sang ảnh xám $G_t$ và tính frame difference trên các cặp khung liên tiếp: $D_t = |G_t - G_{t-1}|$. N khung hợp lệ cho ra N−1 ảnh sai khác (trung vị 29). Phông nền tĩnh cho giá trị gần 0, vùng tay/cánh tay đang chuyển động cho giá trị lớn.
- Nếu video không còn đủ 2 khung hợp lệ thì dùng 1 ảnh sai khác toàn 0.
- Lưu dạng uint8 trong file .npz nén để dùng lại qua mọi epoch, không phải giải mã video nhiều lần.

**Bước 2 – Chuẩn hóa:** do giá trị sai khác thường nhỏ, ảnh được khuếch đại 4 lần, cắt ở 255 rồi đưa về [−1, 1]:
$X_t = 2\cdot\min(4D_t, 255)/255 - 1$.
Sau đó lặp 1 kênh thành 3 kênh giống nhau để dùng được trọng số ImageNet. [−1, 1] cũng chính là khoảng chuẩn hóa gốc của MobileNetV2.

**Bước 3 – Lấy mẫu thời gian:** đưa mọi video về **T = 16** bước:
$i_k = \mathrm{clip}(\mathrm{round}(\mathrm{linspace}(0, n-1, 16)_k \cdot s), 0, n-1)$.
- Khi đánh giá: s = 1, các bước trải đều từ đầu đến cuối video.
- Khi huấn luyện: s ~ U(0,9; 1,1), mô phỏng người ký nhanh hoặc chậm hơn.
- Nếu n < 16 thì một số chỉ số được lặp lại.

**Bước 4 – Tăng cường dữ liệu (chỉ khi huấn luyện):**
- Co giãn thời gian như trên.
- Cắt ngẫu nhiên một vùng vuông chiếm 85–100% diện tích rồi resize lại 224×224. **Cùng một vị trí cắt cho cả 16 bước**, để không tạo chuyển động giả.
- Không dùng color jitter vì đầu vào đã không còn màu.

Tensor đầu vào cuối cùng có kích thước B×16×224×224×3.

**Bước 5 – Kiến trúc mô hình** (TensorFlow/Keras, cùng tinh thần với nhánh motion của Temporal Segment Network / two-stream):

| Khối | Thành phần | Đầu ra |
|---|---|---|
| Đầu vào | 16 ảnh sai khác, 3 kênh, [−1,1] | B×16×224×224×3 |
| Trích xuất đặc trưng | TimeDistributed(MobileNetV2, ImageNet, bỏ top, global avg pool) + Dropout 0,3 | B×16×1280 |
| Tổng hợp thời gian | GlobalAveragePooling1D: $h = \frac{1}{16}\sum_t f_t$ | B×1280 |
| Phân loại | Dropout 0,4 + Dense(100), float32 | B×100 logits → softmax → argmax |

- MobileNetV2 gồm các khối inverted residual và tích chập depthwise separable, nên rất nhẹ. Cùng một bộ trọng số được dùng chung cho cả 16 bước, nên số tham số không tăng theo T.
- Phép trung bình theo thời gian không có tham số học, nên ổn định khi dữ liệu ít.
- Tổng số tham số khoảng **2,39 triệu** (backbone khoảng 2,26 triệu, Dense khoảng 128 nghìn).

**Bước 6 – Huấn luyện:**
- **Mất cân bằng lớp:** mỗi epoch lấy N mẫu có hoàn lại, xác suất chọn một video ∝ 1/n_c (n_c là số video của lớp đó), tương đương WeightedRandomSampler.
- **Loss:** categorical cross-entropy với label smoothing 0,1. **Optimizer:** AdamW, weight decay 1e-4.
- **Giai đoạn 1 – khởi động (3 epoch):** đóng băng backbone, chỉ học phần phân loại, learning rate 1e-3. Mục đích: lớp Dense mới hội tụ trước, tránh gradient lớn phá trọng số ImageNet.
- **Giai đoạn 2 – fine-tune (tối đa 30 epoch):** mở toàn bộ backbone, learning rate 1e-4 giảm dần theo CosineDecay.
- **BatchNorm đóng băng** trong suốt quá trình huấn luyện, vì batch nhỏ (4) và các ảnh trong cùng một video tương quan mạnh nên thống kê batch không đáng tin cậy.
- **Checkpoint:** đánh giá trên toàn bộ tập validation sau mỗi epoch, lưu trọng số khi Macro-F1 validation đạt mức cao nhất mới; early stopping với patience 8. Cuối cùng nạp lại trọng số tốt nhất để đánh giá trên test.
- Batch size 4, seed 42, mixed precision, GPU NVIDIA Tesla T4 (Kaggle).

**Bước 7 – Chỉ số đánh giá:** Top-1 Accuracy, Macro-Precision, Macro-Recall, Macro-F1. Các chỉ số macro là trung bình không trọng số trên đủ 100 lớp; lớp có mẫu số bằng 0 được gán 0. Macro-F1 là chỉ số chính (mỗi lớp có trọng số như nhau, công bằng với lớp ít mẫu) và cũng là tiêu chí chọn checkpoint. Ngoài ra đo thời gian suy luận cho một video.

## D. DỮ LIỆU THỰC NGHIỆM

Tổng cộng 3.866 video, 100 lớp; cả 3 tập đều có đủ 100 lớp.

| Tập | Số video | Tỷ lệ | Ít nhất/lớp | Nhiều nhất/lớp | TB/lớp | Số lớp chỉ có 1 mẫu |
|---|---:|---:|---:|---:|---:|---:|
| Train | 3.077 | 79,59% | 4 | 59 | 30,77 | 0 |
| Validation | 385 | 9,96% | 1 | 7 | 3,85 | 26 |
| Test | 404 | 10,45% | 1 | 8 | 4,04 | 23 |

→ Dữ liệu mất cân bằng mạnh. Trên val/test, chỉ cần sai 1 video ở lớp có 1 mẫu là F1 của lớp đó bằng 0.

## E. KẾT QUẢ (đã kiểm tra lại từ file dự đoán, đây là số chính xác)

**Kết quả tổng:**

| Tập | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 |
|---|---:|---:|---:|---:|
| Validation | 91,17% | 82,32% | 83,39% | 82,12% |
| Test | 94,06% | 86,53% | 86,94% | 86,22% |

- Test: đúng 380/404 video, sai 24.
- Validation: đúng 351/385, sai 34.
- Khoảng cách Accuracy − Macro-F1 trên test là 7,84 điểm %, do lỗi tập trung ở các lớp ít mẫu.
- Test cao hơn validation (F1 +4,10 điểm, Acc +2,89 điểm). Chỉ nên coi đây là dao động giữa hai tập nhỏ, không kết luận rằng mô hình tốt hơn trên dữ liệu mới.

**Diễn biến huấn luyện** (epoch đánh số từ 1; train acc được tính trên mẫu đã lấy cân bằng và tăng cường):

| Epoch | Train acc | Val Macro-F1 | Val Acc | Ghi chú |
|---:|---:|---:|---:|---|
| 1 | 10,66% | 7,31% | 11,43% | backbone đóng băng |
| 2 | 29,16% | 18,05% | 21,82% | |
| 3 | 42,65% | 15,52% | 20,78% | hết giai đoạn khởi động |
| 4 | 46,42% | 48,29% | 57,14% | bắt đầu fine-tune → tăng vọt |
| 5 | 76,63% | 63,03% | 69,35% | |
| 7 | 90,90% | 74,30% | 83,90% | |
| 12 | 97,66% | 80,04% | 88,05% | |
| 18 | 99,25% | 81,09% | 90,65% | |
| 24 | 99,97% | 81,19% | 90,91% | |
| **27** | 99,93% | **82,12%** | **91,17%** | **checkpoint tốt nhất** |
| 33 | 99,97% | 81,49% | 90,65% | epoch cuối |

- Chạy đủ 33 epoch (3 khởi động + 30 fine-tune), early stopping không kích hoạt.
- Từ khoảng epoch 12 trở đi, val Macro-F1 dao động trong khoảng 77–82% trong khi train acc gần 100%. Mô hình đã khớp gần hết dữ liệu huấn luyện, nên việc chọn checkpoint theo validation là cần thiết.

**Phân tích theo lớp (test):**
- 80 lớp có Recall = 1; 64 lớp có Precision = Recall = 1.
- 10 lớp có F1 = 0: Ban đêm, Bàn tay, Cách ly, Có thể, Cơ thể, Nhầm, Sốt, Thương, Vâng lời, Xe máy. Trong đó 9 lớp chỉ có 1 video test; riêng "Có thể" có 2 video và sai cả 2.
- Validation có 11 lớp F1 = 0: Chúng ta, Cơ thể, Ho, Học sinh, Ngón tay, Rẽ phải, Thương, Vâng lời, Xe máy, Xin lỗi, Đẹp. Các lớp **Cơ thể, Thương, Vâng lời, Xe máy sai trên cả validation và test**, nên đây là những ký hiệu khó thật sự với mô hình.

**Các cặp nhầm lẫn trên test** (nhãn thật → dự đoán, số lần): Tôi → Xúc động (2). Mỗi cặp sau xuất hiện 1 lần: Xe máy → Đi, Vâng lời → Mời vào, Thương → Nôn ói, Sốt → Biết, Phía sau → Sử dụng, Ô tô → Xe đạp, Nói → Nghe, Nhầm → Đầu, Mời vào → Chậm lại, Hôm nay → Ô tô, Hôm nay → Có thể, Cơ thể → Chân, Có thể → Xe máy, Có thể → Con gấu, Chậm lại → Đầu, Chậm lại → Chào, Cách ly → Biếu tặng, Bộ y tế → Áp dụng, Bệnh viện → Nhà.
- Lỗi phân tán, hầu hết mỗi cặp chỉ 1 lần.
- Một số cặp có thể giải thích hợp lý: Ô tô↔Xe đạp, Nói↔Nghe (cùng vùng miệng/tai), Cơ thể→Chân, Nhầm→Đầu, Chậm lại→Đầu (chuyển động quanh cùng vùng cơ thể). Khi viết cần ghi rõ đây là **giả thuyết**, chưa được kiểm chứng bằng cách xem video.
- Giới hạn phương pháp liên quan: frame difference là giá trị tuyệt đối nên mất **hướng** chuyển động, và phép trung bình thời gian làm mất **thứ tự** các pha động tác.

**Thời gian suy luận:** 88,67 ms/video trên GPU Tesla T4. Đo với batch 1, đầu vào 1×16×224×224×3, gọi qua tf.function, làm nóng 5 lần, đo 30 lần, lấy trung vị. Số này **chỉ tính phần mạng nơ-ron**, chưa gồm đọc video và tính frame difference (các bước này chỉ là chuyển ảnh xám và trừ ảnh nên chi phí thấp).

**Giới hạn thực nghiệm cần nêu:**
- Chỉ có 1 lần huấn luyện (seed 42).
- Nhiều lớp chỉ có 1 mẫu đánh giá.
- Chưa xác nhận cách chia tập độc lập theo người ký.
- Thời gian đo chưa bao gồm toàn bộ pipeline.

## F. HÌNH CÓ SẴN (đặt chỗ trong bài)

- Hình 1: Sơ đồ pipeline (video → loại khung trống, ảnh xám → frame difference ×4 → lấy 16 bước, 3 kênh, [−1,1] → TimeDistributed MobileNetV2 16×1280 → trung bình thời gian → Dropout + Dense 100 → nhãn). Đặt ở mục tổng quan phương pháp.
- Hình 2: Biểu đồ cột 4 chỉ số trên validation và test.
- Hình 3: Đường val Macro-F1 theo epoch, có đánh dấu epoch 27 và vạch "bắt đầu fine-tune" sau epoch 3.
- Hình 4: Biểu đồ các cặp nhầm lẫn trên test.

## G. CẤU TRÚC GỢI Ý

**1. Mô tả phương pháp**

**2. Kết quả thực nghiệm**

