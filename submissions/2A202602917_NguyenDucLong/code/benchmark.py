"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo (vi phạm bị trừ điểm, RUBRIC mục 3):
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() (hoặc CUDA event) TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
  - chọn và ghi rõ có tính tiền xử lý hay không
"""
from __future__ import annotations
import time
import numpy as np
import torch


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây."""
    for _ in range(warmup):
        fn()
        if sync is not None:
            sync()

    times = []
    for _ in range(iters):
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        times.append((time.perf_counter() - t0) * 1000.0)

    times = np.array(times, dtype=np.float64)
    return {
        "p50": float(np.percentile(times, 50)),
        "p95": float(np.percentile(times, 95)),
        "p99": float(np.percentile(times, 99)),
        "mean": float(np.mean(times)),
        "n": iters,
    }


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).
    Trả về dict có thể ghi thẳng vào sheet `Latency` của results.xlsx.
    """
    dev = torch.device(device if (torch.cuda.is_available() and "cuda" in device) else "cpu")
    model = model.to(dev)
    model.eval()

    sync = torch.cuda.synchronize if dev.type == "cuda" else None
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev)

    if dtype == "fp16":
        model = model.half()
        x = x.half()
        fn = lambda: model(x)
    elif dtype == "amp":
        fn = lambda: torch.cuda.amp.autocast()(model)(x) if dev.type == "cuda" else model(x)
    else:  # fp32
        model = model.float()
        x = x.float()
        fn = lambda: model(x)

    with torch.inference_mode():
        res = bench(fn, warmup=warmup, iters=iters, sync=sync)

    gpu_name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"
    p50 = res["p50"]
    images_per_s = float(batch_size / (p50 / 1000.0)) if p50 > 0 else 0.0

    res.update({
        "gpu": gpu_name,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "images_per_s": images_per_s,
        "torch": torch.__version__,
    })
    return res


def tta_latency(model, k_views: int = 2, **kw) -> dict:
    """Đo độ trễ của TTA K view: so sánh độ trễ thực tế với K * p50."""
    single_res = latency_report(model, **kw)
    batch_size = kw.get("batch_size", 1)
    img_size = kw.get("img_size", 224)
    dtype = kw.get("dtype", "fp32")
    device = kw.get("device", "cuda")
    warmup = kw.get("warmup", 10)
    iters = kw.get("iters", 50)

    dev = torch.device(device if (torch.cuda.is_available() and "cuda" in device) else "cpu")
    sync = torch.cuda.synchronize if dev.type == "cuda" else None
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev)

    def tta_fn():
        for _ in range(k_views):
            _ = model(x)

    with torch.inference_mode():
        tta_res = bench(tta_fn, warmup=warmup, iters=iters, sync=sync)

    tta_res.update({
        "single_p50": single_res["p50"],
        "k_views": k_views,
        "expected_k_p50": single_res["p50"] * k_views,
        "overhead_ratio": tta_res["p50"] / (single_res["p50"] * k_views) if single_res["p50"] > 0 else 1.0,
    })
    return tta_res
