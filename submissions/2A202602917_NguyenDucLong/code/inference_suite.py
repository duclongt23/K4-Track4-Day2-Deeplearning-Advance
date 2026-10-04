"""inference_suite.py - Bước 3 (GUIDE mục 4): so sánh các phương pháp suy luận trên VAL, kèm độ trễ.

Không huấn luyện lại, không chạm test. Đầu vào: cấu hình đã train (có best.pt). Đầu ra: DataFrame đúng cột
của sheet `Inference` và `Latency` trong results.xlsx.

    from inference_suite import run_suite
    inf_df, lat_df = run_suite(cfg_best, ensemble_cfgs=[cfg_a, cfg_b])
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark import latency_report
from inference import (aggregate_views, apply_temperature, fit_temperature, fuse_conv_bn, predict_logits,
                       view_hflip, view_identity)
from train import Config, build_eval_loaders, import_eval, run_dir, softmax_np


def load_model(cfg: Config, device) -> nn.Module:
    from model import build_model
    m = build_model(cfg.backbone, False, 9, 0.0, "scratch").to(device)
    m.load_state_dict(torch.load(run_dir(cfg) / "best.pt", map_location=device))
    return m.eval()


def resize_view(size: int):
    return lambda x: F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


class MultiView(nn.Module):
    """Bọc (model, view) thành một module trả về TRUNG BÌNH XÁC SUẤT: dùng để đo độ trễ thật của TTA/ensemble."""

    def __init__(self, pairs):
        super().__init__()
        self.models = nn.ModuleList([m for m, _ in pairs])
        self.views = [v for _, v in pairs]

    def forward(self, x):
        return torch.stack([F.softmax(m(v(x)), -1) for m, v in zip(self.models, self.views)]).mean(0)


def _metrics(ev, y, probs) -> dict:
    m = ev.compute_metrics(y, probs.argmax(1), probs)
    return {"macro_f1_val": m["macro_f1"], "top1_val": m["top1"], "ece_val": m["ece"]}


def _crossfit_ece(ev, logits, y, seed: int = 0) -> float:
    """ECE sau temperature scaling khi T khớp trên nửa này và đo trên nửa kia (tránh đo trên chính dữ liệu khớp)."""
    idx = np.random.default_rng(seed).permutation(len(y))
    halves = (idx[: len(y) // 2], idx[len(y) // 2:])
    out = []
    for a, b in (halves, halves[::-1]):
        T = fit_temperature(logits[a], y[a])
        p = apply_temperature(logits[b], T)
        out.append(ev.compute_metrics(y[b], p.argmax(1), p)["ece"])
    return float(np.mean(out))


def _lat(module, cfg, device, dtype="fp32", batch=1, iters=100, size=None) -> dict:
    return latency_report(module, batch, size or cfg.img_size, dtype, device.type, warmup=10, iters=iters)


def run_suite(cfg: Config, ensemble_cfgs: list[Config] | None = None, sizes=(224, 256, 288, 320),
              iters: int = 100, latency: bool = True):
    """Chạy I00..I08 trên val. `ensemble_cfgs`: các cấu hình KHÁC (backbone/seed khác) đã train, để thử ensemble."""
    import dataset
    ev = import_eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_df, _ = dataset.load_split(cfg.labels_dir, cfg.fold)
    val_loader, _ = build_eval_loaders(cfg, val_df)
    model = load_model(cfg, device)
    rows, lat_rows = [], []

    def add(exp_id, method, model_desc, k, probs, y, mod=None, dtype="fp32", note="", extra=None):
        row = {"exp_id": exp_id, "method": method, "model": model_desc, "K": k}
        row.update(_metrics(ev, y, probs))
        if extra:
            row.update(extra)
        row["note"] = note
        if latency and mod is not None:
            b1 = _lat(mod, cfg, device, dtype, 1, iters)
            b32 = _lat(mod, cfg, device, dtype, 32, max(30, iters // 3))
            row.update({"p50_ms": b1["p50"], "p95_ms": b1["p95"], "p99_ms": b1["p99"],
                        "throughput_img_s": b32["images_per_s"]})
            lat_rows.append({"config": f"{exp_id} {method}", "gpu": b1["gpu"], "dtype": dtype, "batch": 1,
                             "fused_bn": "fused" in method.lower(), "p50": b1["p50"], "p95": b1["p95"],
                             "p99": b1["p99"], "img_s": b1["images_per_s"], "torch": b1["torch"]})
            lat_rows.append({"config": f"{exp_id} {method}", "gpu": b32["gpu"], "dtype": dtype, "batch": 32,
                             "fused_bn": "fused" in method.lower(), "p50": b32["p50"], "p95": b32["p95"],
                             "p99": b32["p99"], "img_s": b32["images_per_s"], "torch": b32["torch"]})
        rows.append(row)
        return row

    # I00: 1 view (mốc)
    names, y, lg0 = predict_logits(model, val_loader, device, None)
    base = add("I00", "1 view (mốc)", cfg.exp_id, 1, softmax_np(lg0), y, model, note="resize 256 + center crop 224")

    # I01 / I03: TTA lật ngang, gộp xác suất vs gộp logit
    _, _, lg_flip = predict_logits(model, val_loader, device, view_hflip)
    hflip_mod = MultiView([(model, view_identity), (model, view_hflip)]).to(device).eval()
    add("I01", "TTA lật ngang, gộp xác suất", cfg.exp_id, 2, aggregate_views([lg0, lg_flip], "prob"), y, hflip_mod)
    add("I03", "TTA lật ngang, gộp logit", cfg.exp_id, 2, aggregate_views([lg0, lg_flip], "logit"), y, hflip_mod,
        note="cùng chi phí với I01; chỉ khác cách gộp")

    # I02: đa tỉ lệ {224, 256} bằng nội suy từ batch 224 (không đọc lại ảnh)
    ms_sizes = (224, 256)
    ms_logits = [predict_logits(model, val_loader, device, resize_view(s))[2] for s in ms_sizes]
    ms_mod = MultiView([(model, resize_view(s)) for s in ms_sizes]).to(device).eval()
    add("I02", f"TTA đa tỉ lệ {ms_sizes}", cfg.exp_id, len(ms_sizes), aggregate_views(ms_logits, "prob"), y, ms_mod)

    # I04: dò độ phân giải kiểm tra (không đổi tham số, chỉ đổi đầu vào)
    for s in sizes:
        if s == cfg.img_size:
            continue
        try:
            loader_s, _ = build_eval_loaders(replace(cfg, img_size=s), val_df)
            _, ys, lgs = predict_logits(model, loader_s, device, None)
        except Exception as e:  # ViT/Swin cố định kích thước
            print(f"I04 size {s} bỏ qua: {e}")
            continue
        if latency:
            b1 = latency_report(model, 1, s, "fp32", device.type, 10, iters)
        add(f"I04_{s}", f"Độ phân giải kiểm tra {s}", cfg.exp_id, 1, softmax_np(lgs), ys, None,
            note=f"test-res {s}, train-res {cfg.img_size}",
            extra=({"p50_ms": b1["p50"], "p95_ms": b1["p95"], "p99_ms": b1["p99"]} if latency else None))

    # I05: ensemble (trung bình xác suất)
    if ensemble_cfgs:
        members = [model] + [load_model(c, device) for c in ensemble_cfgs]
        probs = [softmax_np(lg0)] + [softmax_np(predict_logits(m, val_loader, device, None)[2]) for m in members[1:]]
        ens_mod = MultiView([(m, view_identity) for m in members]).to(device).eval()
        add("I05", "Ensemble (TB xác suất)", "+".join([cfg.exp_id] + [c.exp_id for c in ensemble_cfgs]),
            len(members), np.mean(probs, 0), y, ens_mod, note="chi phí ~ số mô hình")

    # I07: temperature scaling (T khớp trên val). ECE đo cả in-sample lẫn cross-fit.
    T = fit_temperature(lg0, y)
    add("I07", "Temperature scaling", cfg.exp_id, 1, apply_temperature(lg0, T), y, model,
        note=f"T={T:.3f}; ECE trước {base['ece_val']:.4f}",
        extra={"T": T, "ece_val_crossfit": _crossfit_ece(ev, lg0, y)})

    # I08: gộp BN, FP16, AMP (accuracy phải gần như không đổi)
    fused = fuse_conv_bn(model.cpu()).to(device).eval()
    model.to(device)
    _, _, lg_fused = predict_logits(fused, val_loader, device, None)
    add("I08a", f"Gộp BN vào conv ({getattr(fused, 'n_fused', 0)} cặp)", cfg.exp_id, 1, softmax_np(lg_fused), y, fused)
    if device.type == "cuda":  # FP16/AMP chỉ có nghĩa trên GPU
        half = load_model(cfg, device).half()
        _, _, lg_half = predict_logits(half, val_loader, device, lambda x: x.half())
        add("I08b", "FP16", cfg.exp_id, 1, softmax_np(lg_half), y, model, dtype="fp16",
            note="độ chính xác đo trên model.half(); độ trễ đo trên bản FP16")
        add("I08c", "AMP (autocast)", cfg.exp_id, 1, softmax_np(lg0), y, model, dtype="amp",
            note="độ chính xác lấy từ FP32 (autocast chỉ đổi tốc độ)")

    inf = pd.DataFrame(rows)
    inf["cost_vs_I00"] = inf["p50_ms"] / base["p50_ms"] if "p50_ms" in inf and base.get("p50_ms") else np.nan
    return inf, pd.DataFrame(lat_rows)
