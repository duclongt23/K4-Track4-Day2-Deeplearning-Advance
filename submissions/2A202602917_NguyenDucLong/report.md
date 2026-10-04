# Báo cáo Lab Day 2: Backbone, công thức huấn luyện và suy luận trên DeepWeeds

> KHUNG BÁO CÁO. Mọi chỗ `<điền ...>` phải lấy từ `results.xlsx` hoặc `eval.py` sau khi chạy thật. Không nhập số từ bài báo gốc như kết quả của mình (xem README mục 6). Xoá dòng này trước khi nộp.

## 1. Tóm tắt (≤ 10 dòng)

<điền: bài toán; số backbone/thí nghiệm đã làm; cấu hình tốt nhất; macro-F1 và top-1 test (mean ± std, 3 seed); mức cải thiện so với mốc T00+I00 và có vượt nhiễu không; kết luận chính>

## 2. Dữ liệu và thiết lập

- DeepWeeds, fold 0 chia sẵn. Số ảnh thật: train <điền> / val <điền> / test <điền> (tổng 17.509); giao các tập rỗng; đủ file ảnh. (Dán từ `dataset.check_split`.)
- Phân bố lớp: <chèn `figs/eda_class_distribution.png`>. Tỉ lệ lớp lớn nhất/nhỏ nhất: <điền>. Đối chiếu Table 1 bài báo: <điền nhận xét>.
- Ảnh mẫu và cặp lớp dễ nhầm bằng mắt: <chèn `figs/eda_samples.png`, nhận xét>.
- Kiểm tra pipeline: loss ban đầu <điền> (kỳ vọng ln 9 ≈ 2,197); overfit 1 batch xuống <điền>; ảnh sau augmentation khớp nhãn (`figs/aug_check.png`).
- Chỉ số: macro-F1 (chính), top-1, balanced accuracy, F1 theo lớp, ECE 15 bin; định nghĩa theo README mục 2.2. Chọn mọi thứ trên val; test chạy một lần mỗi seed ở Bước 4.
- Công thức nền, phần cứng, phiên bản thư viện, seed: <điền>.

## 3. So sánh backbone

<bảng từ sheet `Backbones`; biểu đồ macro-F1 theo độ trễ/tham số; nhận xét: hội tụ, quá khớp, thứ hạng so với ImageNet, FLOPs có dự đoán được độ trễ không; backbone được chọn và LÝ DO bằng số liệu>

## 4. Công thức huấn luyện

<bảng từ sheet `Training` với Δ so với T00 và so sánh với nhiễu seed (std T00 = <điền>). Nêu rõ yếu tố nào giúp, yếu tố nào không, "không phân biệt được" khi |Δ| ≤ std; cách tham lam theo trục; kết hợp T13 cộng dồn hay triệt tiêu; giải thích liên hệ slide>

## 5. Suy luận

<bảng từ sheet `Inference` và `Latency`; đường đánh đổi (`figs/tradeoff.png`); ECE trước/sau temperature scaling (T = <điền>, khớp trên val); TTA tăng bao nhiêu điểm, tốn bao nhiêu lần độ trễ; phương pháp nào hợp ngoại tuyến, phương pháp nào hợp thời gian thực. Điều kiện đo: GPU, dtype, warmup, synchronize, số lần đo>

## 6. Cấu hình tốt nhất

- Mô tả đầy đủ để tái lập: <backbone + tag trọng số; các yếu tố huấn luyện; phương pháp suy luận; epochs; seed>.
- Kết quả test (mean ± std, 3 seed) so với mốc: <bảng từ sheet `Final`, kèm output `eval.py score` và `eval.py grade`>.
- Recall/F1 Chinee Apple và Snake Weed: <điền>.
- Ma trận nhầm lẫn: <chèn `figs/confusion_F01_test.png`>; ảnh bị đoán sai: <chèn `figs/errors_F01.png`>; giả thuyết nguyên nhân: <điền>.

## 7. Kết luận và khuyến nghị

- Cấu hình nào tốt nhất, tốt hơn mốc bao nhiêu, có vượt nhiễu không: <điền>.
- Yếu tố đóng góp nhiều nhất (backbone, huấn luyện hay suy luận), dẫn số từ xlsx: <điền>.
- Triển khai robot (ngân sách 30–100 ms/khung): chọn <cấu hình>, p95 batch-1 = <điền> ms, macro-F1 test = <điền>.

## 8. Hạn chế và việc tiếp theo

<số seed ở từng bước; một fold; chia ngẫu nhiên không theo địa điểm nên điểm test có thể lạc quan; giảm bớt do ngân sách GPU (số epoch, ablation trên 1 backbone); thí nghiệm thất bại; lệch phân phối; việc làm tiếp>

## 9. Phụ lục

Danh sách `exp_id` và cấu hình (từ `runs/*/seed*/config.json`), link notebook: <điền>.
