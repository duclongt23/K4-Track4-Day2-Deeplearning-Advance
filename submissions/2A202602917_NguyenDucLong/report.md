# Báo cáo Lab Day 2: Backbone, công thức huấn luyện và suy luận trên DeepWeeds

Sinh viên: Nguyễn Đức Long (2A202602917). Mọi con số lấy từ `results.xlsx`, `eval.py score/grade` và `auto_decisions.json` trong thư mục này. Số trích từ bài báo gốc được ghi rõ là *trích dẫn*.

## 1. Tóm tắt

- Bài toán: phân loại 9 lớp (8 loài cỏ dại + `Negative`) trên DeepWeeds, fold 0 chia sẵn.
- Đã làm: 6 backbone (B01–B06), 12 biến thể công thức huấn luyện + 1 kết hợp (T01–T13) trên ConvNeXt-T, 12 phương pháp suy luận trên val kèm độ trễ, chung kết 3 seed và mốc 3 seed.
- Cấu hình tốt nhất: **ConvNeXt-Tiny** (trọng số `in12k_ft_in1k`) + **TrivialAugmentWide** + suy luận 1 view FP32, 10 epoch.
- Kết quả test (3 seed, mean ± std): **top-1 0,9779 ± 0,0016, macro-F1 0,9716 ± 0,0014**. Mốc (T00 + I00): top-1 0,9745 ± 0,0021, macro-F1 0,9678 ± 0,0033.
- Cải thiện so với mốc: Δ macro-F1 = +0,0037, chỉ hơn std lớn nhất (0,0033) một chút, nên **cải thiện nhỏ và ở mức biên, không chắc chắn**.
- Yếu tố đóng góp nhiều nhất là **backbone cùng trọng số tiền huấn luyện** (macro-F1 val từ 0,66 đến 0,97 giữa các backbone) và **việc có tiền huấn luyện** (từ đầu: 0,29). Các thay đổi còn lại của công thức và suy luận chỉ đổi macro-F1 val khoảng ±0,01, phần lớn xấp xỉ nhiễu.
- Cấu hình thời gian thực: cùng mô hình, 1 view FP32, p95 = 8,9 ms ở batch 1 trên Tesla T4 (FP16: 7,4 ms).

## 2. Dữ liệu và thiết lập

**Chia dữ liệu.** Fold 0 nguyên bản (`train/val/test_subset0.csv`). Kiểm tra bằng `dataset.check_split`: train 10.501, val 3.501, test 3.507 (tổng 17.509); tỉ lệ 60,0/20,0/20,0; giao các cặp tập rỗng; mọi file ảnh tồn tại.

| Lớp | train | val | test |
|---|---|---|---|
| Chinee Apple | 675 | 225 | 226 |
| Lantana | 637 | 213 | 213 |
| Parkinsonia | 618 | 206 | 207 |
| Parthenium | 613 | 204 | 205 |
| Prickly Acacia | 637 | 212 | 213 |
| Rubber Vine | 605 | 202 | 202 |
| Siam Weed | 644 | 215 | 215 |
| Snake Weed | 609 | 203 | 204 |
| Negatives | 5.463 | 1.821 | 1.822 |

Mất cân bằng: lớp lớn nhất/nhỏ nhất trong train là 9,03. `Negatives` chiếm khoảng 52% mỗi tập, nên top-1 bị lớp này kéo cao và macro-F1 là chỉ số chính. Số đếm khớp Table 1 của bài báo (`Negative` 9.106; các loài 1.009–1.125 ảnh) sau khi nhân tỉ lệ 60%/20%/20%. Biểu đồ: `figs/eda_class_distribution.png`; ảnh mẫu: `figs/eda_samples.png`.

**Kiểm tra pipeline (trước khi chạy thật).** Loss ban đầu của ResNet-50 với head mới là 2,171 (kỳ vọng ln 9 ≈ 2,197). Overfit 1 batch 16 ảnh: lần kiểm tra đầu (40 bước, LR 1e-4 cho cả head, đo loss ở `eval()`) chỉ xuống **0,889**, do ô kiểm tra thiết kế yếu chứ không phải lỗi pipeline. Chạy lại với 100 bước, LR head 1e-3, không augmentation: loss **2,196 → 0,085 (bước 25) → 0,006 (bước 50) → 0,003 (bước 75)**; loss cuối **0,0020 ở chế độ train và 0,0020 ở chế độ eval**, tức mô hình học thuộc được batch nhỏ và hai chế độ BatchNorm cho kết quả khớp. (Lần chạy lại này dùng một phiên Colab khác nên chạy sau các thí nghiệm chính; nó chỉ là kiểm tra pipeline, không ảnh hưởng số liệu nào khác.) Ảnh sau augmentation khớp nhãn: `figs/aug_check.png`.

**Chỉ số.** macro-F1 (chính), top-1, balanced accuracy, precision/recall/F1 theo lớp, ECE 15 bin, đúng định nghĩa README mục 2.2; tính bằng `eval.py`. Mọi lựa chọn dùng **val**; test chỉ chạm một lần mỗi seed ở Bước 4 qua `final.finalize` (có chốt chặn trong code).

**Công thức nền `T00`.** AdamW; LR backbone 1e-4, head 1e-3; weight decay 0,05 (0 cho norm/bias); warmup 1 epoch rồi cosine; CE; batch 64; AMP; 10 epoch; `RandomResizedCrop(224)` + lật ngang; val/test resize 256 rồi center-crop 224; chọn checkpoint theo macro-F1 val (hòa lấy epoch sớm).

**Môi trường.** Google Colab, GPU Tesla T4, torch 2.11.0+cu130, timm 1.0.29. Seed 0 cho mọi quét sàng; seed 0, 1, 2 cho T00 và chung kết. Tag trọng số của từng backbone nằm ở bảng mục 3.

## 3. So sánh backbone (1 seed, cùng công thức nền)

| exp_id | Backbone | Tag trọng số | Tham số (M) | GMAC | macro-F1 val | top-1 val | s/epoch | p50 batch-1 (ms, sơ bộ) |
|---|---|---|---|---|---|---|---|---|
| B01 | resnet50 | a1_in1k | 23,5 | 4,13 | 0,7863 | 0,8446 | 45,5 | 7,3 |
| B02 | resnext50_32x4d | a1h_in1k | 23,0 | 4,29 | 0,6616 | 0,7615 | 54,6 | 8,1 |
| B03 | convnext_tiny | in12k_ft_in1k | 27,8 | 4,45 | **0,9664** | **0,9734** | 50,8 | 5,8 |
| B04 | deit_small_patch16_224 | fb_in1k | 21,7 | 4,24 | 0,9454 | 0,9612 | 40,4 | 5,2 |
| B05 | efficientnet_b0 | ra_in1k | 4,0 | 0,38 | 0,7923 | 0,8472 | 40,1 | 7,9 |
| B06 | mobilenetv3_large_100 | ra_in1k | 4,2 | 0,22 | 0,6959 | 0,7818 | 37,1 | 8,7 |

Nhận xét:

- ConvNeXt-T và DeiT-S hội tụ nhanh và cao hơn hẳn (đường cong: `curves/B03_convnext_tiny.png`, `curves/B04_deit_small_patch16_224.png`). ResNet-50, ResNeXt, EfficientNet và MobileNet **chưa hội tụ** sau 10 epoch: train loss và val loss gần nhau và còn giảm (ví dụ ResNet-50: train 0,41, val 0,45 ở epoch 10), tức là **underfit**, không phải overfit.
- **Không tách được ảnh hưởng của kiến trúc và của trọng số.** Ba tag yếu (`a1_in1k`, `a1h_in1k`, `ra_in1k`) có công thức tiền huấn luyện khác tag của ConvNeXt (`in12k_ft_in1k`, huấn luyện trước trên ImageNet-12k) và DeiT (`fb_in1k`). Với 10 epoch và LR 1e-4, các trọng số này tinh chỉnh chậm hơn. Vì vậy kết quả **không** cho phép kết luận ResNet kém ConvNeXt về kiến trúc; đây là so sánh các *gói (kiến trúc, trọng số, công thức)*. Thứ hạng ở đây cũng có thể khác thứ hạng trên ImageNet (slide trang 41–42); chưa tách được nguyên nhân.
- FLOPs không dự đoán độ trễ: EfficientNet-B0 (0,38 GMAC) và MobileNetV3 (0,22 GMAC) chậm hơn ConvNeXt-T (4,45 GMAC) ở batch 1 (7,9 và 8,7 ms so với 5,8 ms), chưa biết lý do tại sao. Số độ trễ ở bước này đo sơ bộ (50 lần, FP32).
- **Chọn ConvNeXt-T** đi tiếp vì macro-F1 val cao nhất, với độ trễ sơ bộ thấp (5,8 ms, xấp xỉ DeiT-S 5,2 ms); DeiT-S được giữ làm thành viên ensemble ở Bước 3 vì thuộc họ khác.

## 4. Công thức huấn luyện (ConvNeXt-T, 1 seed, khác T00 đúng một yếu tố)

Nhiễu seed: std (ddof=1) của macro-F1 val của T00 qua 3 seed (0,9664; 0,9697; 0,9688) là **0,0017**.

| exp_id | Trục | Khác T00 | macro-F1 val | Δ so với T00 | Δ so với nhiễu |
|---|---|---|---|---|---|
| T00 | nền | (nền) | 0,9664 | | |
| T01 | A. Khởi tạo | đóng băng backbone | 0,8488 | −0,1176 | rất rõ |
| T02 | A. Khởi tạo | từ đầu | 0,2909 | −0,6755 | rất rõ |
| T03 | B. Augmentation | + ColorJitter | 0,9654 | −0,0010 | không phân biệt được |
| T04 | B. Augmentation | TrivialAugmentWide | **0,9735** | **+0,0071** | ≈ 4× std |
| T05 | B. Augmentation | Mixup (α=1) | 0,9652 | −0,0012 | không phân biệt được |
| T06 | B. Augmentation | CutMix (α=1) | 0,9656 | −0,0008 | không phân biệt được |
| T07 | C. Loss | label smoothing 0,1 | 0,9693 | +0,0029 | ≈ 1,7× std |
| T08 | C. Loss | focal γ=2 | 0,9620 | −0,0044 | thấp hơn, ≈ 2,5× std |
| T09 | C. Loss | CE trọng số lớp | 0,9559 | −0,0105 | thấp hơn rõ |
| T10 | D. Cân bằng mẫu | sampler cân bằng lớp | 0,9690 | +0,0026 | ≈ 1,5× std |
| T11 | E. LR | cùng LR 1e-4 cho head | 0,9675 | +0,0012 | không phân biệt được |
| T12 | F. Chính quy hoá | EMA 0,998 | 0,9669 | +0,0005 | không phân biệt được |
| T13 | kết hợp | T04 + T07 + T10 | 0,9678 | +0,0014 | không phân biệt được |

Nhận xét (đường cong từng thí nghiệm ở `curves/T0x_convnext_tiny.png`):

- **Tiền huấn luyện là yếu tố lớn nhất.** Đóng băng backbone mất 0,12 điểm F1; huấn luyện từ đầu sau 10 epoch chỉ đạt top-1 0,56 (gần tỉ lệ lớp `Negatives` ≈ 52%) và macro-F1 0,29, tức gần như chưa học được các lớp hiếm (`curves/T02_convnext_tiny.png`). Kết quả này phù hợp với slide về dữ liệu ít và thiên kiến quy nạp yếu.
- **Augmentation:** chỉ TrivialAugmentWide giúp rõ (+0,0071). ColorJitter, Mixup, CutMix không phân biệt được với nhiễu (hơi thấp hơn). Với ảnh cỏ dại, lớp được xác định bởi những chi tiết nhỏ (hình dạng lá), nên cắt dán (CutMix) và trộn ảnh (Mixup) có thể làm mất đặc trưng của đối tượng, hoặc làm tăng nhiễu cục bộ. Đây là giả thuyết, chưa kiểm chứng riêng.
- **Loss:** các loss "chống mất cân bằng" (focal, CE trọng số lớp) **làm giảm** macro-F1 val (−0,0044 và −0,0105). Label smoothing tăng nhẹ (+0,0029) nhưng chỉ ≈ 1,7× nhiễu.
- **Sampler cân bằng, LR, EMA:** không phân biệt được hoặc chỉ nhích nhẹ.
- **Kết hợp không cộng dồn.** Quy tắc chọn tự động ghép T04, T07, T10 (mỗi trục lấy yếu tố thắng). Kết quả T13 = 0,9678, **thấp hơn T04 một mình** (0,9735): hiệu ứng triệt tiêu hoặc phần "thắng" của T07 và T10 chủ yếu là nhiễu.
- **Hạn chế:** mỗi yếu tố chỉ thử 1 seed và có 12 yếu tố, nên có thể chọn nhầm một yếu tố chỉ "thắng" do may mắn. Chung kết 3 seed cho thấy T04 chỉ cải thiện +0,0037 trên test, nhỏ hơn nhiều so với +0,0071 ở 1 seed.

## 5. Suy luận (trên val, mô hình T04; độ trễ Tesla T4, warmup 10, `cuda.synchronize`, 100 lần đo, p50/p95/p99)

| exp_id | Phương pháp | macro-F1 val | top-1 val | ECE val | p50 / p95 / p99 batch-1 (ms) | ảnh/s (batch 32) | chi phí vs I00 |
|---|---|---|---|---|---|---|---|
| I00 | 1 view (mốc) | 0,9735 | 0,9803 | 0,0086 | 6,0 / 8,9 / 10,9 | 244 | 1,00 |
| I01 | TTA lật ngang, gộp xác suất | 0,9739 | 0,9806 | 0,0074 | 11,8 / 14,1 / 15,4 | 122 | 1,97 |
| I03 | TTA lật ngang, gộp logit | 0,9734 | 0,9803 | 0,0085 | 11,7 / 13,3 / 17,1 | 122 | 1,95 |
| I02 | TTA đa tỉ lệ {224, 256} | 0,9762 | 0,9826 | 0,0069 | 12,5 / 13,9 / 17,2 | 107 | 2,09 |
| I04 | Test ở 256 (train 224) | 0,9762 | 0,9829 | 0,0062 | 7,4 / 10,4 / 11,4 | | 1,23 |
| I04 | Test ở 288 | 0,9744 | 0,9806 | 0,0079 | 8,1 / 8,5 / 10,1 | | 1,36 |
| I04 | Test ở 320 | 0,9702 | 0,9763 | 0,0077 | 10,2 / 11,1 / 12,7 | | 1,70 |
| I05 | Ensemble T04 + DeiT-S (B04) | 0,9750 | 0,9823 | 0,0133 | 14,9 / 20,1 / 20,4 | 137 | 2,49 |
| I07 | Temperature scaling, T = 1,369 | 0,9735 | 0,9803 | 0,0052 (in-sample); 0,0089 (cross-fit) | 7,7 / 11,0 / 12,2 | 245 | 1,29 |
| I08a | Gộp BN vào conv | 0,9735 | 0,9803 | 0,0086 | 6,1 / 9,0 / 10,5 | 245 | 1,02 |
| I08b | FP16 | 0,9737 | 0,9806 | 0,0085 | 5,6 / 7,4 / 10,6 | 807 | 0,94 |
| I08c | AMP (autocast) | 0,9735 | 0,9803 | 0,0086 | 7,9 / 9,2 / 13,3 | 641 | 1,32 |

(Dòng I04 không đo thông lượng batch 32.) Nhận xét:

- **TTA lật ngang** gần như không giúp (+0,0004 F1, dưới nhiễu 0,0017) trong khi **tốn gần 2× độ trễ** (đúng K = 2). Gộp xác suất và gộp logit khác nhau không đáng kể. TTA đa tỉ lệ và đổi độ phân giải test lên 256 tăng macro-F1 val +0,0027 (trên nhiễu 0,0017 nhưng chỉ 1 mô hình, 1 seed, và chỉ trên val). Độ phân giải 256 gần "miễn phí" hơn TTA (chi phí 1,23× so với 2,09×), phù hợp với ý FixRes (slide trang 68): test ở độ phân giải cao hơn lúc train có thể tốt hơn; nhưng tăng thêm lên 288, 320 thì giảm.
- **Ensemble** (+0,0015 F1, dưới nhiễu) làm **ECE tệ hơn** (0,0133) và tốn 2,5× độ trễ: không đáng.
- **Hiệu chuẩn:** temperature scaling khớp trên val (T = 1,369, tức mô hình hơi quá tự tin) giảm ECE val từ 0,0086 xuống 0,0052 khi đo trên chính val, nhưng khi khớp T trên một nửa val và đo trên nửa kia (cross-fit) ECE là 0,0089, **không tốt hơn** trước (0,0086). Vì vậy quy tắc tự động **không dùng TS** ở chung kết (phần ECE trước/sau trên test vì thế không có bản "uncal" riêng; xem mục 8).
- **Gộp BN:** ConvNeXt dùng LayerNorm, không có cặp conv-BN để gộp (0 cặp), nên dòng I08a không mang thông tin; nó chỉ xác nhận hàm gộp không phá mô hình. Phương pháp này hợp với ResNet/EfficientNet, nhưng chưa đo trong bài.
- **FP16 và AMP:** FP16 giữ nguyên độ chính xác (0,9737 so với 0,9735) và tăng thông lượng batch 32 từ 244 lên 807 ảnh/s (3,3×), batch 1 hơi nhanh hơn (5,6 so với 6,0 ms). **AMP (autocast) ở batch 1 chậm hơn FP32** (7,9 so với 6,0 ms), đúng cảnh báo ở slide trang 73: không giả định FP16/AMP luôn nhanh hơn.
- **Độ nhiễu của phép đo độ trễ.** I07 chạy cùng mô hình với I00 nhưng đo ra 7,7 ms so với 6,0 ms, nên chênh lệch p50 dưới khoảng 2 ms ở batch 1 trên Tesla T4 (GPU dùng chung trên Colab) chưa đủ để kết luận.
- **Hợp ngoại tuyến và thời gian thực.** Ngoại tuyến (không ràng buộc thời gian): TTA đa tỉ lệ hoặc test ở 256 cho macro-F1 val cao nhất (0,9762). Thời gian thực: không tốn thêm gì, chọn 1 view và FP16 nếu cần thông lượng. Dữ liệu của tôi ủng hộ kết luận của slide rằng TTA và ensemble hợp ngoại tuyến hơn.
- Biểu đồ đánh đổi: `figs/tradeoff.png`.

## 6. Cấu hình tốt nhất và kết quả test

**Cấu hình `F01` (để tái lập):** `convnext_tiny`, tag `in12k_ft_in1k`, `TrivialAugmentWide` + lật ngang + `RandomResizedCrop(224)`, CE, AdamW (1e-4 backbone, 1e-3 head, wd 0,05 không áp dụng cho norm/bias), warmup 1 epoch + cosine, batch 64, 10 epoch, AMP, seed 0, 1, 2; suy luận 1 view FP32, không temperature scaling. Chọn hoàn toàn dựa trên val.

**Kết quả test, chạy đúng một lần mỗi seed (`eval.py score`, 3507 ảnh):**

| Chỉ số | F01 (mean ± std, 3 seed) | Mốc T00 + I00 (3 seed) |
|---|---|---|
| top-1 | **0,9779 ± 0,0016** | 0,9745 ± 0,0021 |
| macro-F1 | **0,9716 ± 0,0014** | 0,9678 ± 0,0033 |
| balanced accuracy | 0,9719 ± 0,0018 | 0,9722 ± 0,0025 |
| ECE (15 bin) | 0,0099 ± 0,0003 | 0,0112 ± 0,0022 |
| macro-F1 val | 0,9713 ± 0,0022 | 0,9683 ± 0,0017 |

- Δ macro-F1 test = +0,0037, std lớn hơn trong hai nhóm = 0,0033: **chỉ hơn 1,1× std**. Trên val Δ = +0,0030 (std 0,0022). Hướng cải thiện nhất quán nhưng nhỏ; chênh lệch val/test của F01 chỉ 0,0003.
- **Balanced accuracy không cải thiện** (0,9719 so với 0,9722), nên phần top-1 tăng đến từ lớp `Negative` chứ không từ các loài hiếm.
- Hai lớp khó (recall test, trung bình 3 seed): Chinee Apple **0,950 ± 0,014** (mốc T00 0,953 ± 0,025), Snake Weed **0,940 ± 0,033** (mốc T00 0,944 ± 0,016): không khác mốc. Số tham khảo từ bài báo (ResNet-50, *trích dẫn*): 88,5% và 88,8%; so sánh chỉ mang tính tham chiếu vì khác điều kiện (bài báo: 5 fold, khoảng 100 epoch, augmentation mạnh, "weighted average accuracy"). Top-1 của bài báo (trích dẫn): ResNet-50 95,7%, Inception-v3 95,1%.
- `eval.py grade` (đề xuất, giảng viên xác nhận): I1 = 7/7, I2 = 4/5, I3 = 4/4, I4b = 1/1, I5 = 2/2 (p95 8,9 ms ≤ 100 ms, đo đúng cách); I4a chưa chấm được vì không có bản chưa/đã temperature scaling (xem mục 8). Tổng đã chấm 18/19.

**Ma trận nhầm lẫn** (cộng 3 seed, `figs/confusion_F01_test.png`): Chinee Apple → Snake Weed 13/678 (1,9%) và Snake Weed → Chinee Apple 11/612 (1,8%); số trích dẫn của bài báo là 3,4% và 4,1%. Tuy nhiên lỗi nhiều nhất về số lượng **liên quan đến `Negative`**: `Negative` → Prickly Acacia 22 ảnh, Chinee Apple → `Negative` 20, Snake Weed → `Negative` 18, `Negative` → Rubber Vine 14.

**Phân tích lỗi bằng ảnh** (`figs/errors_F01.png`, ảnh Chinee Apple ↔ Snake Weed bị đoán sai của seed 0): phần lớn là ảnh tối hoặc nhiều bóng đổ, lá nhỏ lẫn trong đất. Giả thuyết: Độ tối và âm sắc màu của hình ảnh khá giống nhau, khiến mô hình khó phân biệt. Ngoài ra, kỹ thuật chia dữ liệu chỉ dựa vào thời gian có thể khiến các loài khó phân biệt với nhau và với nền, do chúng có thể xuất hiện trong cùng một lô đất hoặc điều kiện ánh sáng.

## 7. Kết luận và khuyến nghị

- **Cấu hình tốt nhất:** ConvNeXt-T `in12k_ft_in1k` + TrivialAugmentWide + 1 view, macro-F1 test 0,9716 ± 0,0014. Tốt hơn mốc +0,0037, **chỉ vượt nhiễu ở mức biên**.
- **Yếu tố đóng góp nhiều nhất:** backbone cùng trọng số tiền huấn luyện (chênh lệch tới 0,3 macro-F1 val giữa các backbone) và tiền huấn luyện (từ đầu: −0,68). Công thức huấn luyện (trong số đã thử) đem lại tối đa khoảng +0,007 val và +0,004 test; suy luận (TTA, ensemble, độ phân giải) khoảng +0,003 val. Tức là chọn backbone/trọng số trước, sau đó mới chỉnh công thức.
- **Triển khai robot (ngân sách 30–100 ms/khung):** dùng F01 với 1 view; p95 = 8,9 ms (FP32) hoặc 7,4 ms (FP16) ở batch 1 trên Tesla T4, thấp hơn nhiều so với ngân sách. Có thể thử test ở 256 (p95 10,4 ms) vì chi phí nhỏ và cho +0,0027 F1 val; chưa kiểm chứng trên test, nên đây chỉ là đề xuất. Không khuyến nghị TTA lật ngang hay ensemble (tốn 2–2,5× độ trễ mà gần như không tăng độ chính xác).

## 8. Hạn chế và trung thực

- **Số seed:** quét backbone và ablation chỉ 1 seed; chỉ T00 và chung kết có 3 seed. Nhiều chênh lệch ở mục 4 nằm trong khoảng 1–2× nhiễu và không nên diễn giải mạnh. Yếu tố "thắng" T04 (+0,0071 ở 1 seed) cho cải thiện test nhỏ hơn (+0,0037) ở 3 seed: dấu hiệu của chọn lọc theo may mắn khi thử nhiều yếu tố.
- **Một fold** (fold 0), chia ngẫu nhiên không theo địa điểm: ảnh gần nhau về thời gian/vị trí có thể rơi vào cả train và test, nên điểm test có thể **lạc quan** so với khi gặp địa điểm mới.
- **Cắt giảm do ngân sách GPU:** 10 epoch (bài báo khoảng 100); ablation chỉ trên ConvNeXt-T; không dò LR, weight decay hay độ dài huấn luyện; cách tham lam theo trục (thứ tự trục có thể ảnh hưởng).
- **Backbone bị bất lợi bởi trọng số:** xem mục 3; chưa chạy thêm cho ResNet-50/EfficientNet với tag hoặc LR khác để tách hai yếu tố.
- **Cách chọn cấu hình:** các lựa chọn (yếu tố huấn luyện, TTA, temperature scaling) được áp dụng bằng một quy tắc cố định trên val (mã ở `code/auto_pipeline.py`). Quy tắc suy luận chỉ xét TTA lật ngang và temperature scaling; TTA đa tỉ lệ và test ở 256 (tốt hơn trên val) không được đưa vào chung kết, và tôi không chạy lại test để thử chúng (test chỉ chạy một lần mỗi seed).
- **I4a:** temperature scaling không được chọn nên không có cặp "trước/sau" trên test; điểm I4a không chấm được. ECE test của F01 (0,0099) thấp hơn mốc (0,0112) nhưng đó là hai cấu hình huấn luyện khác nhau, không phải hiệu ứng của temperature scaling.
- **Đường cong F01 và T00:** mỗi `exp_id` một ảnh; với chạy nhiều seed, ảnh là của seed cuối.
- **Độ trễ:** đo trên GPU dùng chung (Tesla T4 trên Colab) nên có nhiễu khoảng ±2 ms ở batch 1; chỉ tính model, không tính tiền xử lý và nhập/xuất ảnh.
- **Lệch phân phối:** chưa đánh giá trên ảnh mùa khác, thiếu sáng hay mờ; T khớp trên val có thể không còn đúng khi đổi miền.
- **Việc tiếp theo nếu có thêm thời gian:** nhiều fold; thêm seed cho ablation; thử ResNet-50 với trọng số khác hoặc LR cao hơn; đánh giá TTA đa tỉ lệ / test ở 256 trong một vòng chung kết riêng; Grad-CAM cho cặp Chinee Apple ↔ Snake Weed.

## 9. Phụ lục

- Danh sách `exp_id`: B01–B06 (backbone), T00–T13 (huấn luyện), I00–I08 (suy luận), F01 (chung kết). Bảng đầy đủ nằm ở `results.xlsx` (các sheet `Backbones`, `Training`, `Inference`, `Final`, `PerClass`, `Latency`, `Summary`).
- Cấu hình từng lần chạy: `runs/<exp_id>/seed<k>/config.json` (trên Google Drive, không đưa vào git); quyết định tự động: `auto_decisions.json`.
- Notebook chạy lại: xem `README.md`.
