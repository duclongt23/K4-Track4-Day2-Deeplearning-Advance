# Bài nộp Lab Day 2: DeepWeeds (2A202602917, Nguyễn Đức Long)

## Chạy lại

Notebook Colab (GPU T4): [lab_day2.ipynb](https://colab.research.google.com/github/duclongt23/K4-Track4-Day2-Deeplearning-Advance/blob/main/submissions/2A202602917_NguyenDucLong/code/lab_day2.ipynb)

Thứ tự chạy: Runtime → Run all (ô "TỰ ĐỘNG" chạy Bước 2–4 theo quy tắc cố định trong `code/auto_pipeline.py`; các ô thủ công nằm ở phụ lục cuối notebook). Ô cài đặt tự mount Google Drive, clone repo, tải dữ liệu (kiểm MD5) và ghi mọi kết quả vào `MyDrive/lab_day2_deepweeds/`. Nếu Colab ngắt, chạy lại 3 ô đầu rồi chạy tiếp: thí nghiệm nào đã xong sẽ được bỏ qua.

| Bước | Ô trong notebook | File code |
|---|---|---|
| 0. EDA, kiểm tra split, kiểm tra pipeline | EDA, loss ban đầu ≈ ln 9, overfit 1 batch, ảnh sau augmentation | `dataset.py` |
| 1. 6 backbone (B01–B06) | `experiments.backbone_cfgs` | `model.py`, `train.py` |
| 2. Công thức huấn luyện (T00–T13) | `experiments.training_cfgs`, `combo_cfg` | `losses.py`, `train.py` |
| 3. Suy luận + độ trễ (I00–I08) | `inference_suite.run_suite` | `inference.py`, `benchmark.py`, `inference_suite.py` |
| 4. Chung kết, test một lần mỗi seed | `final.finalize` + `eval.py score/grade` | `final.py`, `eval.py` (gốc, không sửa) |
| 5. `results.xlsx`, báo cáo | `build_results.build_xlsx` | `build_results.py` |

Chạy riêng một thí nghiệm: `python train.py --set exp_id=B01 backbone=resnet50 seed=0`.

Kiểm tra tự viết (focal γ=0 ≡ CE, CutMix, gộp BN, temperature scaling, nhóm tham số, EMA...): `cd code && python -m unittest test_components -v`.

## Cấu hình

- Dataset: DeepWeeds, **fold 0** (`train/val/test_subset0.csv` nguyên bản, không sửa). Seed không đổi cách chia.
- Công thức nền `T00`: AdamW, LR backbone 1e-4 / head 1e-3, weight decay 0,05 (không áp dụng cho norm/bias), warmup 1 epoch + cosine, CE, batch 64, AMP, 10 epoch (giống nhau mọi thí nghiệm).
- Seed: ablation 1 seed (0); chung kết và mốc `T00` dùng seed 0, 1, 2.
- Môi trường: Google Colab, GPU Tesla T4, torch 2.11.0+cu130, Python 3.13; phiên bản timm: 1.0.29.
- Tag trọng số `timm` của từng backbone: xem cột `tag trọng số` trong sheet `Backbones` của `results.xlsx`.

## Sản phẩm

`results.xlsx`, `report.md`, `curves/` (mỗi `exp_id` một ảnh), `predictions/` (chung kết + mốc, mỗi seed), `code/`. Dataset và checkpoint không được commit.
