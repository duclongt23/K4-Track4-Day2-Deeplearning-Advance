"""final.py - Bước 4: chạy TEST đúng một lần cho mỗi (exp_id, seed) và ghi file dự đoán nộp bài.

Dùng sau khi `train.run(cfg)` đã huấn luyện xong cấu hình đó (có best.pt). Một lần gọi:
  - tính logit val và test với phương pháp suy luận đã chốt (1 view, hoặc TTA lật ngang),
  - (tuỳ chọn) khớp nhiệt độ T trên VAL rồi áp sang test (không dùng thông tin test: quy tắc S4),
  - ghi predictions/<exp_id>_seed<k>_test.csv và <exp_id>_seed<k>_val.csv,
    kèm predictions/<exp_id>uncal_seed<k>_test.csv (chưa temperature scaling, để eval.py grade chấm I4a).

Giao thức: mỗi (exp_id, seed) chỉ được chạy test một lần. Gọi lần hai sẽ báo lỗi (trừ khi force=True, và
khi đó phải ghi chuyện này vào báo cáo; RUBRIC mục 3).
"""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import torch

from train import Config, build_eval_loaders, import_eval, pred_path, run_dir, softmax_np


def finalize(cfg: Config, tta: str = "none", temperature: bool = False, force: bool = False) -> dict:
    """tta: "none" (1 view) | "hflip" (trung bình logit của ảnh gốc và ảnh lật ngang)."""
    import dataset
    from inference import apply_temperature, fit_temperature, predict_logits, view_hflip, view_identity
    from model import build_model

    ev = import_eval()
    out_dir = run_dir(cfg)
    guard = out_dir / "test_done.json"
    if guard.exists() and not force:
        raise RuntimeError(f"{cfg.exp_id} seed{cfg.seed}: đã chạy test rồi ({guard}). "
                           "Chạy test lần hai để chọn kết quả là vi phạm quy tắc; muốn ghi đè phải force=True.")
    if tta not in ("none", "hflip"):
        raise ValueError(f"tta phải là none | hflip, nhận {tta!r}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.backbone, False, 9, 0.0, "scratch").to(device)
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    model.eval()

    _, val_df, test_df = dataset.load_split(cfg.labels_dir, cfg.fold)
    val_loader, test_loader = build_eval_loaders(cfg, val_df, test_df)
    views = [view_identity] + ([view_hflip] if tta == "hflip" else [])

    def logits_of(loader):
        per_view = [predict_logits(model, loader, device, v) for v in views]
        names, y = per_view[0][0], per_view[0][1]
        return names, y, np.mean([lg for _, _, lg in per_view], axis=0)  # trung bình LOGIT (so sánh với prob: mục I03)

    v_names, v_y, v_logits = logits_of(val_loader)
    t_names, t_y, t_logits = logits_of(test_loader)

    T = fit_temperature(v_logits, v_y) if temperature else 1.0   # T chỉ khớp trên VAL
    v_probs = apply_temperature(v_logits, T)
    t_probs = apply_temperature(t_logits, T)
    info = {"exp_id": cfg.exp_id, "seed": cfg.seed, "tta": tta, "temperature": temperature, "T": T,
            "val_ece_before": ev.compute_metrics(v_y, v_logits.argmax(1), softmax_np(v_logits))["ece"],
            "val_ece_after": ev.compute_metrics(v_y, v_probs.argmax(1), v_probs)["ece"],
            "val_macro_f1": ev.compute_metrics(v_y, v_probs.argmax(1), v_probs)["macro_f1"]}

    if temperature:  # bản chưa hiệu chuẩn để chấm I4a
        ev.save_predictions(pred_path(replace(cfg, exp_id=cfg.exp_id + "uncal"), "test"),
                            t_names, t_y, softmax_np(t_logits))
    ev.save_predictions(pred_path(cfg, "test"), t_names, t_y, t_probs)
    ev.save_predictions(pred_path(cfg, "val"), v_names, v_y, v_probs)
    np.save(out_dir / "test_logits.npy", t_logits)
    guard.write_text(json.dumps(info, indent=2))
    print(f"[final] {cfg.exp_id} seed{cfg.seed} tta={tta} T={T:.3f} -> {pred_path(cfg, 'test')}")
    return info
