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
    """Đo độ trễ forward (chỉ model, không tính tiền xử lý) với đầu vào ngẫu nhiên (B, 3, S, S).

    dtype: "fp32" | "amp" (autocast fp16) | "fp16" (model.half()). Model truyền vào được sao chép nên
    không bị đổi dtype/thiết bị. Trả về dict ghi thẳng được vào sheet `Latency` của results.xlsx.
    """
    import copy
    dev = torch.device(device if (torch.cuda.is_available() and "cuda" in device) else "cpu")
    model = copy.deepcopy(model).to(dev).eval()
    sync = torch.cuda.synchronize if dev.type == "cuda" else None
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev)

    if dtype == "fp16":
        model, x = model.half(), x.half()
        fn = lambda: model(x)
    elif dtype == "amp":
        def fn():
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                return model(x)
    elif dtype == "fp32":
        model = model.float()
        fn = lambda: model(x)
    else:
        raise ValueError(f"dtype phải là fp32 | amp | fp16, nhận {dtype!r}")

    with torch.inference_mode():
        res = bench(fn, warmup=warmup, iters=iters, sync=sync)

    res.update({
        "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU",
        "dtype": dtype, "batch": batch_size, "img_size": img_size,
        "images_per_s": float(batch_size / (res["p50"] / 1000.0)) if res["p50"] > 0 else 0.0,
        "torch": torch.__version__,
    })
    return res


def tta_latency(model, k_views: int = 2, batch_size: int = 1, img_size: int = 224, dtype: str = "fp32",
                device: str = "cuda", warmup: int = 10, iters: int = 50) -> dict:
    """Độ trễ của TTA K view (K lần forward liên tiếp) so với K * p50 của 1 view (slide trang 63)."""
    single = latency_report(model, batch_size, img_size, dtype, device, warmup, iters)
    dev = torch.device(device if (torch.cuda.is_available() and "cuda" in device) else "cpu")
    import copy
    m = copy.deepcopy(model).to(dev).eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev)
    if dtype == "fp16":
        m, x = m.half(), x.half()
    sync = torch.cuda.synchronize if dev.type == "cuda" else None

    def tta_fn():
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=(dtype == "amp" and dev.type == "cuda")):
            for _ in range(k_views):
                m(x)

    with torch.inference_mode():
        res = bench(tta_fn, warmup=warmup, iters=iters, sync=sync)
    res.update({"single_p50": single["p50"], "k_views": k_views, "expected_k_p50": single["p50"] * k_views,
                "ratio_vs_k_times_single": res["p50"] / (single["p50"] * k_views) if single["p50"] > 0 else float("nan")})
    return res
