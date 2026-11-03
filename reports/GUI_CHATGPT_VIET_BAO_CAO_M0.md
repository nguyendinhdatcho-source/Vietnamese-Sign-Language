# GÓI THÔNG TIN ĐỂ CHATGPT VIẾT BÁO CÁO PHƯƠNG PHÁP M0

Tệp này là tài liệu bàn giao, không phải bản báo cáo để nộp. Có thể tải riêng tệp này lên ChatGPT; các thông tin cần thiết từ mã nguồn, notebook tiền xử lý chung và kết quả thực nghiệm đã được tập hợp bên dưới. Không cần gửi phần demo hoặc toàn bộ dữ liệu video để viết hai phần báo cáo.

---

## A. YÊU CẦU VIẾT — DÙNG NHƯ PROMPT CHO CHATGPT

Tôi đang làm báo cáo đồ án nhận dạng ngôn ngữ ký hiệu Việt Nam. Nhóm tôi có bốn phương pháp để so sánh, và tôi phụ trách phương pháp frame difference kết hợp MobileNetV2 và trung bình đặc trưng theo thời gian, ký hiệu M0. Hãy dùng hồ sơ kỹ thuật và kết quả thực tế trong tệp này để viết mới phần báo cáo của tôi bằng tiếng Việt.

### Phạm vi và hình thức đầu ra

- Chỉ viết hai phần chính: **Mô tả phương pháp** và **Kết quả thực nghiệm**. Không viết phần demo, tổng quan toàn bộ đồ án, hướng dẫn chạy mã hay giới thiệu các phương pháp khác.
- Chỉ trình bày M0. Không đưa các biến thể khác của nhánh chuyển động vào bảng so sánh. Chưa có kết quả của ba phương pháp còn lại của nhóm, nên không tự tạo bảng so sánh hoặc khẳng định M0 đứng đầu.
- Không dùng từ “baseline” để gọi phương pháp. Dùng “phương pháp”, “mô hình” hoặc “M0”. Không tự gán đóng góp mới về mặt nghiên cứu.
- Phần **Mô tả phương pháp** dài khoảng **2–3 trang A4**, theo cỡ chữ Times New Roman 12 và giãn dòng 1,15. Có thể dùng 4–5 tiểu mục, một sơ đồ ngắn, một bảng kiến trúc và các công thức thật sự cần thiết. Độ dài trang chỉ là mục tiêu trình bày, không phải lý do kéo dài nội dung.
- Độ dài phần kết quả tùy lượng phân tích cần thiết; ưu tiên giải thích có căn cứ hơn là liệt kê nhiều bảng gần giống nhau.

### Cách viết tôi mong muốn

Hãy viết như một phần hoàn chỉnh trong báo cáo đồ án đại học: rõ ràng, có mạch lập luận, giải thích được ý tưởng, cách thực hiện và ý nghĩa của kết quả. Không biến báo cáo thành bản mô tả từng dòng mã hoặc nhật ký kiểm tra tệp. Tránh những đoạn nói về việc bạn đã đọc, kiểm chứng hoặc tổng hợp số liệu như thế nào; các việc đó chỉ là cơ sở để viết chính xác.

Trong phần phương pháp, cần nối được ba ý: vì sao dùng ảnh sai khác để nhấn mạnh thay đổi giữa các khung hình; cách dùng mạng 2D tiền huấn luyện để trích xuất đặc trưng tại từng bước; và cách tổng hợp các đặc trưng thành dự đoán cho video. Diễn đạt lợi ích như lý do thiết kế, không biến thành kết luận thực nghiệm khi chưa có thí nghiệm đối chứng.

Trong phần thực nghiệm, trình bày điều kiện đánh giá trước, sau đó phân tích kết quả tổng thể, quá trình học, lỗi theo lớp và tốc độ. Mỗi bảng hoặc hình cần có đoạn giải thích giúp người đọc hiểu điều gì đáng chú ý. Phân biệt rõ quan sát trực tiếp từ số liệu với giả thuyết về nguyên nhân.

Các bước đã có trong phần tiền xử lý chung của nhóm không cần trình bày lại. Chỉ dẫn chiếu ngắn và mô tả các điều chỉnh riêng của M0 theo mục C. Không cần đưa toàn bộ bảng đối chiếu trong tài liệu bàn giao này vào báo cáo.

### Giới hạn về dữ kiện

- Chỉ dùng số liệu đã cung cấp. Không tự thêm kết quả huấn luyện, kiểm định thống kê, khoảng tin cậy, thông số phần cứng hay số liệu của phương pháp khác.
- Không khẳng định khả năng chạy thời gian thực từ thời gian suy luận riêng của mạng.
- Không khẳng định đánh giá độc lập theo người ký; chưa có signer ID được xác minh.
- Không mô tả ảnh sai khác như một cách tách hoàn toàn bàn tay, phông nền hoặc danh tính người ký. Không kết luận mô hình chỉ học chuyển động mà hoàn toàn không chứa dấu vết hình dạng.
- Không kết luận fine-tune là “cần thiết” hoặc giúp cải thiện bao nhiêu so với một mô hình đóng băng đã huấn luyện đầy đủ: hiện chỉ có đường cong của một lần chạy hai giai đoạn.
- Không sử dụng tên kiến trúc khác để gọi mô hình nếu mã nguồn không triển khai đầy đủ kiến trúc đó. Có thể giải thích trực tiếp thiết kế CNN 2D dùng chung trọng số và pooling thời gian.
- Nếu cần nhắc tài liệu bên ngoài, hãy kiểm tra nguồn và dẫn nguồn thật. Không tự tạo tài liệu tham khảo. Bản báo cáo vẫn có thể được viết chỉ từ hồ sơ thực nghiệm bên dưới.

### Điểm cần xử lý trung thực khi viết

Notebook tiền xử lý chung và lần chạy M0 hiện tại **không dùng cùng cách chia tập**. Chi tiết được nêu ở mục F. Hãy ghi một đoạn ngắn trong thiết lập thực nghiệm để phân biệt, giữ nguyên kết quả đã đo của M0. Không tự thay số mẫu, không gán kết quả M0 cho cách chia mới và không để vấn đề này lấn át phần mô tả phương pháp. Nếu nhóm muốn so sánh trực tiếp sau này thì cần chạy các phương pháp trên cùng danh sách video.

Hãy viết ngay bản hoàn chỉnh dựa trên các dữ kiện này. Những thông tin chưa có thì lược bỏ hoặc nêu giới hạn đúng mức; không cần hỏi lại các thông tin đã nằm trong tệp.

---

## B. HỒ SƠ KỸ THUẬT ĐÃ XÁC ĐỊNH

### B1. Bài toán và định danh lần chạy

| Nội dung | Dữ kiện |
|---|---|
| Bài toán | Phân loại video ký hiệu rời rạc, mỗi video có một nhãn |
| Ngôn ngữ / số lớp | Ngôn ngữ ký hiệu Việt Nam / 100 lớp |
| Định danh dùng trong báo cáo | M0; frame difference + MobileNetV2 + temporal average pooling |
| Thư mục kết quả gốc | vsl_motion_work/runs/m0_diff_avg |
| Cấu hình đầu vào | mode = diff; T = 16; size = 224 |
| Backbone / mô-đun thời gian | mobilenetv2 / avg |
| Thư viện mô hình | TensorFlow/Keras |
| Phạm vi | Video đã được chia theo từng ký hiệu; không phải dịch câu hoặc nhận dạng liên tục |

### B2. Thứ tự xử lý thực tế của M0

1. Đọc toàn bộ video, loại từng khung đồng màu theo tiêu chí độ lệch chuẩn không gian từng kênh ≤ 1 trên thang uint8.
2. Resize khung còn lại về 224 × 224 bằng INTER_AREA rồi chuyển sang ảnh xám.
3. Tính sai khác tuyệt đối trên các cặp kế tiếp trong chuỗi còn lại. Với N khung hợp lệ, có N − 1 ảnh sai khác.
4. Lưu ảnh sai khác uint8 trong trường `diff` của tệp `.npz` nén. Đây là bước tiền tính, không tính lại khi đọc mỗi batch.
5. Khi huấn luyện hoặc đánh giá, lấy 16 ảnh từ chuỗi sai khác đã tiền tính; tăng cường thời gian chỉ được dùng khi train.
6. Khuếch đại giá trị sai khác với hệ số 4, cắt ngưỡng và đưa tạm về [0,1]. Khi train, crop cùng một vùng cho toàn clip rồi resize bằng INTER_LINEAR nếu cần.
7. Đưa về [−1,1], lặp ảnh một kênh thành ba kênh; đầu vào là float32 theo thứ tự kênh cuối.
8. Trích xuất đặc trưng từng bước bằng MobileNetV2, trung bình theo thời gian, rồi phân loại.

Trong mã tiền tính hiện tại, chương trình còn tạo một biểu diễn chuyển động khác trong cùng tệp; M0 chỉ đọc trường `diff`. Không đưa nhánh còn lại vào báo cáo M0, và không dùng tổng thời gian chương trình tiền tính để suy ra riêng chi phí frame difference.

Nếu không có cặp khung hợp lệ, mã nguồn tạo một ảnh sai khác bằng 0 để giữ định dạng. Hồ sơ dùng cho gói bàn giao này không có thống kê số video phải dùng trường hợp dự phòng đó.

### B3. Công thức và quy tắc đầu vào

Với Gₜ là ảnh xám của bước t trong chuỗi sau khi lọc:

```text
Dₜ = |Gₜ − Gₜ₋₁|
Xₜ = 2 × min(4Dₜ, 255) / 255 − 1
```

Đây là sai khác tuyệt đối về cường độ ảnh, không phải vector dịch chuyển. Ba kênh đầu vào giống nhau; không có ba nguồn thông tin màu độc lập.

Với n ảnh sai khác, chỉ số lấy mẫu trong mã:

```text
idx = clip(round(linspace(0, n − 1, 16) × s), 0, n − 1)
s ~ Uniform(0.9, 1.1) khi train; s = 1 khi đánh giá
```

M0 không thêm độ lệch ngẫu nhiên của điểm bắt đầu. Khi n < 16, các vị trí lấy đều có thể bị lặp. Khi s > 1, các vị trí vượt chuỗi được cắt về cuối chuỗi. Crop train giữ xấp xỉ 85–100% diện tích ảnh vuông, cùng vị trí cho mọi bước. Không có color jitter hoặc lật ngang trong nhánh M0 hiện tại.

### B4. Kiến trúc chính xác

```text
Input                                  (B, 16, 224, 224, 3)
TimeDistributed(MobileNetV2)            (B, 16, 1280)
  include_top=False
  weights='imagenet'
  pooling='avg' theo không gian
Dropout(0.3)                           (B, 16, 1280)
GlobalAveragePooling1D                 (B, 1280)
Dropout(0.4)                           (B, 1280)
Dense(100, dtype='float32')             (B, 100), logits
Softmax + argmax khi dự đoán            Một nhãn/video
```

Các bước dùng chung một MobileNetV2, không phải 16 backbone có trọng số khác nhau. Với fₜ là đặc trưng của bước t:

```text
h = (1/16) × Σₜ fₜ
```

Phép pooling thời gian không có tham số học và không phân biệt thứ tự các vector đầu vào. Mô hình này không chứa mạng hồi quy, tích chập thời gian hoặc tích chập 3D.

Không cần đưa số tham số vào báo cáo nếu chưa kiểm tra trực tiếp bằng `model.count_params()` hoặc cách kiểm tra tương đương. Không có phép đo FLOPs hay bộ nhớ thực tế trong hồ sơ này.

### B5. Huấn luyện

| Thiết lập | Giá trị |
|---|---|
| Batch size / seed | 4 / 42 |
| Hàm mất mát | CategoricalCrossentropy(from_logits=True, label_smoothing=0.1) |
| Bộ tối ưu | AdamW |
| Weight decay | 0.0001 ở cả hai giai đoạn |
| Giai đoạn 1 | 3 epoch; backbone đóng băng; learning rate 0.001 |
| Giai đoạn 2 | Tối đa 30 epoch; mở backbone, giữ BatchNorm đóng băng |
| Learning rate giai đoạn 2 | Khởi đầu 0.0001; CosineDecay |
| Dropout | 0.3 sau backbone; 0.4 trước Dense |
| Chọn checkpoint | Macro-F1 validation cao nhất qua các epoch |
| Early stopping | patience = 8 |
| Thiết bị ghi nhận | GPU NVIDIA Tesla T4 trên Kaggle |
| Mixed precision | Được bật khi chạy GPU; logits giữ float32 |
| Số lần chạy dùng để báo cáo | Một lần, seed 42 |

Label smoothing thực tế: y′_c = 0.9y_c + 0.1/100; nhãn đúng = 0.901, các nhãn khác = 0.001.

Lấy mẫu cân bằng sử dụng trọng số mỗi video là 1/n_c và lấy có hoàn lại. Tổng xác suất theo lớp bằng nhau về kỳ vọng, không bảo đảm số lượt của các lớp giống hệt nhau trong mỗi epoch. Triển khai TensorFlow tên `BalancedIndices`, tương đương về nguyên tắc với bộ lấy mẫu cân bằng của notebook chung. Không cần giải thích lại nội dung này dài trong phần phương pháp riêng.

Với 3.077 lượt lấy mẫu mỗi epoch, batch size 4 và drop_remainder=True, có 769 batch đầy đủ tương ứng 3.076 lượt mẫu. Đây không phải 3.076 video phân biệt.

### B6. Chỉ số và cách đo thời gian

Accuracy, Macro-Precision, Macro-Recall và Macro-F1 được tính bằng scikit-learn. Các chỉ số macro lấy trung bình trên đủ nhãn 0–99, với zero_division=0. Macro-F1 là trung bình F1 từng lớp, không phải trung bình điều hòa của Macro-Precision và Macro-Recall.

Mã đo thời gian dùng tf.function với tensor giả batch size 1, hình dạng (1,16,224,224,3); làm nóng 5 lần, đo 30 lần, lấy trung vị. Mỗi lần gọi có chuyển đầu ra về NumPy để chờ tính toán hoàn tất. Giá trị được lưu là thời gian riêng của mạng, chưa có giải mã video hoặc tiền xử lý. Không có số đo độ trễ toàn pipeline hay phép so sánh tốc độ với ba phương pháp còn lại.

---

## C. ĐỐI CHIẾU VỚI TIỀN XỬ LÝ CHUNG CỦA NHÓM

Nguồn đã đọc: `VSL_EDA_Preprocessing_TinhGon.ipynb`, đặc biệt các phần 3.3–3.8. Đây là notebook EDA và tiền xử lý, không huấn luyện mô hình.

| Nội dung | Đã có trong notebook chung | Điểm cần viết riêng cho M0 |
|---|---|---|
| Ánh xạ nhãn | label_mapping.pkl, chuẩn hóa Unicode NFC, giữ ID nhất quán | Chỉ dẫn chiếu; không giải thích lại |
| EDA và lọc video lỗi | Khảo sát phân bố lớp, kiểm tra chất lượng, loại video không hợp lệ | Không chép lại EDA vào phương pháp riêng |
| Khung đồng màu | Video toàn khung đồng màu bị loại; nếu chỉ có một phần khung đồng màu thì gắn cờ và vẫn giữ video | M0 loại từng khung đồng màu trước khi tính sai khác; đây là xử lý bổ sung |
| Biểu diễn | Ảnh RGB | M0 chuyển ảnh xám rồi tạo chuỗi sai khác tuyệt đối |
| Kích thước và độ dài | 224 × 224, 16 frame | Giữ quy ước; nhấn mạnh M0 lấy 16 ảnh sai khác sau khi đã tính trên các cặp kế tiếp |
| Chuẩn hóa | Chia 255 rồi chuẩn hóa RGB bằng mean/std ImageNet | M0 khuếch đại sai khác ×4, cắt 255, chuẩn hóa [−1,1], lặp ba kênh |
| Lấy mẫu thời gian | linspace; ép int; có độ lệch start ngẫu nhiên khi train; chuỗi ngắn bổ sung frame cuối | M0 dùng round, không thêm start; chuỗi ngắn lặp các chỉ số lấy đều |
| Speed augmentation | [0.9,1.1], chỉ train | Không giải thích lại nguyên lý; chỉ nêu áp dụng lên chuỗi sai khác |
| Random crop | 85–100% diện tích; cùng vùng trên toàn clip | Không giải thích lại nguyên lý; M0 thực hiện trên ảnh sai khác |
| Color jitter | Có brightness/contrast/saturation khi train | M0 không dùng color jitter |
| Cân bằng lớp | WeightedRandomSampler, 1/n_c, có hoàn lại | Cùng nguyên tắc, triển khai BalancedIndices trong TensorFlow; không mô tả lại thành đóng góp riêng |
| Thứ tự tensor | PyTorch: (B,16,3,224,224) | Keras: (B,16,224,224,3); chỉ nhắc ngắn nếu cần |
| Chia tập | Train/validation chia theo nhóm; giữ public/private test | Không khớp lần chạy M0; xem mục F |

Nguyên tắc khi viết: có thể dùng một câu dẫn chiếu phần tiền xử lý chung, sau đó tập trung vào tạo ảnh sai khác, chuẩn hóa riêng và mạng phân loại. Không tuyên bố toàn bộ đầu ra DataLoader RGB đã chuẩn hóa của notebook chung được đưa trực tiếp vào M0, vì M0 tạo đầu vào chuyển động theo nhánh riêng.

---

## D. CÁCH ĐỌC KẾT QUẢ VÀ GIỚI HẠN DIỄN GIẢI

- Accuracy test là 380/404, tương ứng khoảng 94.06%. Macro-F1 test khoảng 86.22%; mỗi lớp có trọng số ngang nhau nên khác cách tổng hợp của Accuracy.
- Có 23 lớp test chỉ một video. Một lỗi ở lớp một mẫu khiến Recall và F1 của lớp bằng 0. Không nên khái quát rằng mô hình đã xử lý tốt hoặc thất bại ổn định cả lớp chỉ từ một mẫu.
- Mười lớp có F1 test bằng 0; chín lớp có một video, riêng “Có thể” có hai video. Đây là bằng chứng cụ thể cho ảnh hưởng của các lớp ít mẫu.
- Checkpoint tốt nhất ở epoch 27, không phải epoch 33. Các chỉ số validation/test trong bảng là của checkpoint tốt nhất.
- Macro-F1 validation cuối warm-up và sau fine-tune có biến động rõ. Có thể mô tả diễn biến, nhưng không suy ra quan hệ nhân quả đã được kiểm chứng bằng một thí nghiệm đối chứng riêng.
- Accuracy train cuối gần 100%, trong khi validation ngừng cải thiện. Có thể nhận xét dấu hiệu khớp mạnh dữ liệu huấn luyện, nhưng không lấy accuracy train trừ Macro-F1 validation để tính “khoảng quá khớp”.
- Test cao hơn validation là một quan sát trên hai tập khác nhau. Không viết rằng test cao hơn chứng minh tổng quát hóa tốt hơn một cách chắc chắn.
- Chỉ có một seed; chưa đo biến thiên giữa các lần chạy. Không có đánh giá độc lập theo người ký hoặc kiểm định ý nghĩa thống kê.
- Không sử dụng các khoảng bootstrap từ bản so sánh cũ: tập nhãn lấy trung bình của hàm bootstrap đó thay đổi theo mẫu lặp, khác với Macro-F1 chính luôn tính đủ 100 lớp.
- Không có thống kê số khung đồng màu đã loại trong bộ nguồn số liệu được kiểm chứng cho gói này. Đừng suy ra số video/khung bị loại từ các đoạn diễn giải trong bản nháp khác.

---

## E. SƠ ĐỒ CÓ THỂ DÙNG KHI VIẾT

```text
Video theo quy ước dữ liệu của nhóm
    ↓
Loại từng khung đồng màu → chuyển ảnh xám
    ↓
Sai khác tuyệt đối của các cặp kế tiếp
    ↓
Lấy 16 bước chuyển động → khuếch đại ×4 → chuẩn hóa [−1,1] → lặp 3 kênh
    ↓
TimeDistributed(MobileNetV2–ImageNet): 16 × 1.280 đặc trưng
    ↓
Dropout 0,3 → trung bình theo thời gian: vector 1.280 chiều
    ↓
Dropout 0,4 → Dense 100 → Softmax → Nhãn ký hiệu
```

Sơ đồ là tóm tắt kiến trúc, không phải hướng dẫn thực thi tăng cường dữ liệu. Thứ tự chi tiết các thao tác trên dữ liệu được nêu ở B2.

---

## F. KHÁC BIỆT CÁCH CHIA DỮ LIỆU CẦN GIỮ ĐÚNG

### F1. Kết quả được lưu trong notebook chung

- Dữ liệu ban đầu: 3.875 video train có nhãn, 1.630 public test và 2.859 private test.
- Notebook ghi nhận loại 9 video train do tiêu chí chất lượng/nhãn, còn 3.866 video.
- Chia 3.866 video thành **3.092 train và 774 validation**, seed 42; validation khoảng 20,02%.
- Các nhóm được nối theo chuỗi pixel có cùng SHA-256 hoặc cùng lớp và cùng tên gốc sau khi bỏ hậu tố `_1`, `_2`, …; chọn nhóm cho validation theo quy tắc tham lam trong từng lớp.
- Notebook ghi nhận số nhóm và hash giao nhau giữa train/validation bằng 0. Điều này không xác nhận độc lập theo người ký hoặc kiểm tra toàn bộ trùng lặp train–public/private test.
- Public/private test giữ nguyên, không có nhãn trong notebook; chỉ kiểm đếm và đọc thử batch, chưa huấn luyện hoặc chấm kết quả.

### F2. Cách chia thực tế của lần chạy M0 đang báo cáo

- **3.077 train, 385 validation, 404 test**, tổng 3.866 video có nhãn.
- Cả ba tập đủ 100 lớp. Danh sách nằm trong `vsl_motion_work/splits.csv`.
- Công cụ make_splits.py đọc các thư mục đã chia sẵn; nó không tự tái tạo phép chia theo nhóm của notebook chung.
- Không đủ bằng chứng để khẳng định hai danh sách là cùng cách chia hoặc kết quả M0 đã được chạy trên 3.092/774.
- Không chuyển các kết quả test 94,06% Accuracy và 86,22% Macro-F1 sang public/private test. Không đổi tên 404 video test thành public test nếu chưa xác minh.

### F3. Cách viết đề nghị

Viết ngắn trong phần thiết lập thực nghiệm rằng kết quả hiện có của M0 được đo trên cách chia 3.077/385/404, khác cách chia trong notebook tiền xử lý chung; các phép so sánh trực tiếp cần thống nhất danh sách video. Giữ nguyên các chỉ số đã đo. Không tự tạo một lần thực nghiệm mới để làm cho số liệu khớp.

---

## G. NGUỒN SỐ LIỆU VÀ CÁC BẢNG CHI TIẾT

Các dữ kiện kỹ thuật lấy từ `precompute_motion.py`, `train_motion.py`, `make_splits.py`; dữ kiện tiền xử lý chung lấy từ nội dung và đầu ra đã lưu trong `VSL_EDA_Preprocessing_TinhGon.ipynb`.

Kết quả M0 lấy từ `metrics.json`, `history.csv`, `train.log`, các tệp dự đoán, xác suất và ma trận nhầm lẫn của `vsl_motion_work/runs/m0_diff_avg`. Khi tạo gói bàn giao, dự đoán nhãn và xác suất đã được đối chiếu, các chỉ số được tính lại với đủ 100 nhãn và kiểm tra khớp tệp metrics. Việc này không thực hiện huấn luyện lại và không cần được kể lại trong bản báo cáo cuối.

Các bảng tự động bên dưới là dữ liệu đầu vào để viết, không bắt buộc đưa toàn bộ vào báo cáo.

### G1. Quy mô các tập của M0


| Tập | Video | Lớp | Ít nhất/lớp | Nhiều nhất/lớp | Trung bình/lớp | Lớp chỉ một mẫu |

|---|---:|---:|---:|---:|---:|---:|

| train | 3077 | 100 | 4 | 59 | 30.77 | 0 |

| val | 385 | 100 | 1 | 7 | 3.85 | 26 |

| test | 404 | 100 | 1 | 8 | 4.04 | 23 |


### G2. Kết quả tổng thể của checkpoint tốt nhất


| Tập | Accuracy (%) | Macro-P (%) | Macro-R (%) | Macro-F1 (%) | Đúng / tổng | Sai |

|---|---:|---:|---:|---:|---:|---:|

| val | 91.17 | 82.32 | 83.39 | 82.12 | 351/385 | 34 |

| test | 94.06 | 86.53 | 86.94 | 86.22 | 380/404 | 24 |


Giá trị chính xác trước khi làm tròn:


```json
{
  "val": {
    "top1": 0.9116883116883117,
    "macro_f1": 0.8211968440056677,
    "macro_precision": 0.8232301587301588,
    "macro_recall": 0.8338571428571429
  },
  "test": {
    "top1": 0.9405940594059405,
    "macro_f1": 0.8622281607281608,
    "macro_precision": 0.8653452380952382,
    "macro_recall": 0.8693809523809523
  },
  "model_ms_per_clip": 88.67191250101314,
  "device": "GPU"
}
```


### G3. Các mốc học tập


| Mốc | Giá trị |
|---|---|

| Số epoch thực tế | 33 = 3 warm-up + 30 fine-tune |

| Val Macro-F1 epoch 1 | 7.3137% |

| Val Macro-F1 epoch 2 | 18.0522% |

| Val Macro-F1 epoch 3 | 15.5224% |

| Val Macro-F1 epoch 4 | 48.2889% |

| Val Macro-F1 epoch 27 | 82.1197% |

| Val Macro-F1 epoch 33 | 81.4852% |

| Accuracy train cuối | 99.9675% |

| Loss train cuối | 0.842867196 |

| Thời gian mạng | 88.67191250 ms/clip |


Epoch trong CSV gốc bên dưới bắt đầu từ 0; khi viết hoặc vẽ hình phải cộng 1. Accuracy train tính trên các lượt lấy mẫu huấn luyện có augmentation; không phải một lần chấm độc lập toàn bộ train ở chế độ đánh giá.


```csv
epoch,accuracy,loss,val_macro_f1,val_top1
0,0.1066319867968559,4.4871039390563965,0.0731374580135084,0.1142857142857142
1,0.2916124761104584,3.325080394744873,0.1805223972347042,0.2181818181818181
2,0.4265279471874237,2.827963352203369,0.1552237654245043,0.2077922077922078
3,0.4642392694950104,2.6671338081359863,0.4828894093236198,0.5714285714285714
4,0.7662549018859863,1.7404102087020874,0.6302762734341681,0.6935064935064935
5,0.864109218120575,1.4723997116088867,0.6187622890525367,0.7298701298701299
6,0.9089726805686952,1.3327205181121826,0.7429583749583749,0.8389610389610389
7,0.945058524608612,1.2068496942520142,0.762054252936606,0.8571428571428571
8,0.9522106647491456,1.1621818542480469,0.7562075702075701,0.8675324675324675
9,0.9678153395652772,1.110578536987305,0.7042317580080737,0.8077922077922078
10,0.973016917705536,1.076964020729065,0.7521706561412443,0.8415584415584415
11,0.9765929579734802,1.0611003637313845,0.8003581418581419,0.8805194805194805
12,0.9892717599868774,1.0015705823898315,0.8048936276142159,0.8831168831168831
13,0.9863458871841432,1.0016875267028809,0.7906841214341215,0.8909090909090909
14,0.9899219870567322,0.9751869440078736,0.7740733433233434,0.8909090909090909
15,0.9921976327896118,0.9537230730056764,0.7980929903429904,0.8961038961038961
16,0.9918725490570068,0.9460216164588928,0.7818950983656867,0.8883116883116883
17,0.9925227761268616,0.93893963098526,0.8109355133170923,0.9064935064935064
18,0.9983745217323304,0.9069566130638124,0.8020882467205998,0.9012987012987012
19,0.9990246891975404,0.8932819962501526,0.7906224886224886,0.9012987012987012
20,0.9990246891975404,0.8902259469032288,0.8019409201909202,0.8961038961038961
21,0.9967490434646606,0.8824672698974609,0.7751769064269064,0.8961038961038961
22,0.99934983253479,0.8764246106147766,0.7977823140911378,0.9012987012987012
23,0.999674916267395,0.8645730018615723,0.8118670496170496,0.9090909090909092
24,0.9990246891975404,0.8615865707397461,0.8061673604173605,0.903896103896104
25,1.0,0.8552520871162415,0.8020113236436766,0.903896103896104
26,0.99934983253479,0.8514413833618164,0.8211968440056677,0.9116883116883115
27,0.999674916267395,0.8486007452011108,0.8013806046894283,0.8987012987012987
28,0.999674916267395,0.8468086123466492,0.8000055777555778,0.9012987012987012
29,0.99934983253479,0.8446293473243713,0.8129086322174558,0.903896103896104
30,0.99934983253479,0.8434641361236572,0.8144996686320215,0.9064935064935064
31,0.999674916267395,0.8428835868835449,0.814950218082571,0.9064935064935064
32,0.999674916267395,0.8428671956062317,0.814851982984336,0.9064935064935064
```


### G4. Thống kê lớp trên test


| Thống kê | Giá trị |
|---|---:|

| Lớp có Recall = 1 | 80 |

| Lớp có Precision = Recall = 1 | 64 |

| Lớp có F1 = 0 | 10 |


Các lớp F1 = 0:


| Lớp | Số video test | Số đúng |
|---|---:|---:|

| Ban đêm | 1 | 0 |

| Bàn tay | 1 | 0 |

| Cách ly | 1 | 0 |

| Có thể | 2 | 0 |

| Cơ thể | 1 | 0 |

| Nhầm | 1 | 0 |

| Sốt | 1 | 0 |

| Thương | 1 | 0 |

| Vâng lời | 1 | 0 |

| Xe máy | 1 | 0 |


### G5. Toàn bộ các hướng nhầm trên test


Bảng dưới liệt kê đủ lỗi ngoài đường chéo, không chỉ 20 cặp trong tệp top-confusions. Nhầm có hướng: A → B không đồng nghĩa B → A. Tổng số video trong bảng phải bằng 24.


| Nhãn thật | Nhãn dự đoán | Số video |
|---|---|---:|

| Tôi | Xúc động | 2 |

| Ăn | Mua | 1 |

| Ban đêm | Chậm lại | 1 |

| Bàn tay | Rau | 1 |

| Bệnh viện | Nhà | 1 |

| Bộ y tế | Áp dụng | 1 |

| Cách ly | Biếu tặng | 1 |

| Chậm lại | Chào | 1 |

| Chậm lại | Đầu | 1 |

| Có thể | Con gấu | 1 |

| Có thể | Xe máy | 1 |

| Cơ thể | Chân | 1 |

| Hôm nay | Có thể | 1 |

| Hôm nay | Ô tô | 1 |

| Mời vào | Chậm lại | 1 |

| Nhầm | Đầu | 1 |

| Nói | Nghe | 1 |

| Ô tô | Xe đạp | 1 |

| Phía sau | Sử dụng | 1 |

| Sốt | Biết | 1 |

| Thương | Nôn ói | 1 |

| Vâng lời | Mời vào | 1 |

| Xe máy | Đi | 1 |


### G6. Những gì chưa có để tránh tự thêm kết luận


Chưa có kết quả ba phương pháp còn lại; chưa có chạy lặp nhiều seed; chưa có đánh giá theo người ký; chưa có đo độ trễ toàn pipeline, FLOPs hoặc bộ nhớ; chưa có kết quả M0 trên cách chia 3.092/774 của notebook chung. Những nội dung này có thể được nêu ngắn như giới hạn, không được biến thành số liệu thực nghiệm.
