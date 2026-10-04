"""build_results.py - gom log thật thành results.xlsx (GUIDE mục 6.1). Không nhập số bằng tay.

Nguồn số liệu:
  - runs/<exp_id>/seed<k>/result.json  (do train.run ghi) cho sheet Backbones / Training
  - DataFrame của inference_suite.run_suite cho sheet Inference / Latency
  - predictions/*.csv + eval.py cho sheet Final / PerClass (test) -> khớp số giảng viên tính lại
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import CLASS_NAMES
from experiments import BACKBONES, TRAIN_AXES
from train import import_eval

HARD = ("Chinee Apple", "Snake Weed")


def collect_runs(runs_dir: str | Path) -> pd.DataFrame:
    rows = []
    for f in sorted(Path(runs_dir).glob("*/seed*/result.json")):
        r = json.loads(f.read_text())
        r["config"] = json.loads((f.parent / "config.json").read_text())
        rows.append(r)
    return pd.DataFrame(rows)


def _f1_hard(r) -> str:
    f1 = r["val_f1_per_class"]
    return " / ".join(f"{n} {f1[CLASS_NAMES.index(n)]:.3f}" for n in HARD)


def sheet_backbones(runs: pd.DataFrame) -> pd.DataFrame:
    notes = {eid: n for eid, _, n in BACKBONES}
    d = runs[runs.exp_id.str.startswith("B")]
    return pd.DataFrame({
        "exp_id": d.exp_id, "backbone": d.backbone, "tag trọng số": d.weights_tag,
        "#tham số (M)": d.params_M, "GMAC": d.gmacs, "độ phân giải": d.img_size, "epoch": d.epochs, "seed": d.seed,
        "macro-F1 val": d.val_macro_f1, "top-1 val": d.val_top1, "thời gian train/epoch (s)": d.time_per_epoch_s,
        "độ trễ batch-1 p50 (ms, sơ bộ)": d.latency_b1_p50_ms, "best epoch": d.best_epoch,
        "ghi chú": d.exp_id.map(notes),
    }).sort_values("exp_id")


def sheet_training(runs: pd.DataFrame, noise_std: float | None = None) -> pd.DataFrame:
    d = runs[runs.exp_id.str.startswith("T") & (runs.seed == 0)].copy()
    base = d.loc[d.exp_id == "T00", "val_macro_f1"]
    base = float(base.iloc[0]) if len(base) else np.nan
    axis = {k: v[0] for k, v in TRAIN_AXES.items()}
    diff = {k: v[1] for k, v in TRAIN_AXES.items()}
    out = pd.DataFrame({
        "exp_id": d.exp_id, "backbone": d.backbone,
        "trục thay đổi": d.exp_id.map(axis).fillna("nền / kết hợp"),
        "khác T00 ở điểm nào": d.exp_id.map(diff).fillna("(xem ghi chú)"),
        "seed": d.seed, "macro-F1 val": d.val_macro_f1, "top-1 val": d.val_top1,
        "Δ macro-F1 so với T00": d.val_macro_f1 - base,
        "F1 lớp khó (val)": d.apply(_f1_hard, axis=1),
        "ghi chú": "1 seed",
    })
    if noise_std is not None and np.isfinite(noise_std):
        out["|Δ| > std seed?"] = np.where(out["Δ macro-F1 so với T00"].abs() > noise_std, "có", "không phân biệt được")
        out.loc[out.exp_id == "T00", "|Δ| > std seed?"] = "-"
    return out.sort_values("exp_id")


def _group(ev, pattern: str, ref_csv: str, what: str):
    return ev.load_group(pattern, ref_csv, ref_what=what)


def sheet_final(ev, pred_dir, labels_dir, ids: dict[str, str]):
    """ids: {exp_id: mô tả cấu hình}. Mỗi exp_id có <id>_seed<k>_val.csv và <id>_seed<k>_test.csv."""
    pred_dir, labels_dir = Path(pred_dir), Path(labels_dir)
    rows, perclass = [], []
    names = CLASS_NAMES
    for eid, desc in ids.items():
        gv = _group(ev, str(pred_dir / f"{eid}_seed*_val.csv"), str(labels_dir / "val_subset0.csv"), "val")
        gt = _group(ev, str(pred_dir / f"{eid}_seed*_test.csv"), str(labels_dir / "test_subset0.csv"), "test")
        for i, p in enumerate(gt.preds):
            v = next(m for q, m in zip(gv.preds, gv.metrics) if q.seed == p.seed)
            m = gt.metrics[i]
            rows.append({"exp_id": eid, "cấu hình": desc, "seed": p.seed, "macro-F1 val": v["macro_f1"],
                         "macro-F1 test": m["macro_f1"], "top-1 test": m["top1"], "ECE test": m["ece"]})
        s = gt.summary
        rows.append({"exp_id": eid, "cấu hình": desc, "seed": f"mean ± std ({len(gt.preds)} seed)",
                     "macro-F1 val": ev.fmt(*gv.summary["macro_f1"]), "macro-F1 test": ev.fmt(*s["macro_f1"]),
                     "top-1 test": ev.fmt(*s["top1"]), "ECE test": ev.fmt(*s["ece"])})
        pm, ps = s["precision"]; rm, rs = s["recall"]; fm, fs = s["f1"]
        for c, n in enumerate(names):
            perclass.append({"cấu hình": eid, "lớp": n, "số ảnh test": int(gt.metrics[0]["support"][c]),
                             "precision": ev.fmt(pm[c], ps[c], 3), "recall": ev.fmt(rm[c], rs[c], 3),
                             "F1": ev.fmt(fm[c], fs[c], 3)})
    return pd.DataFrame(rows), pd.DataFrame(perclass)


def sheet_inference(inf: pd.DataFrame) -> pd.DataFrame:
    keep = {"exp_id": "exp_id", "method": "phương pháp", "model": "mô hình/checkpoint", "K": "K (view/mô hình)",
            "macro_f1_val": "macro-F1 val", "top1_val": "top-1 val", "ece_val": "ECE val",
            "p50_ms": "p50 batch-1 (ms)", "p95_ms": "p95 (ms)", "p99_ms": "p99 (ms)",
            "throughput_img_s": "thông lượng batch-32 (ảnh/s)", "cost_vs_I00": "chi phí tương đối vs I00",
            "T": "T", "ece_val_crossfit": "ECE val sau TS (cross-fit)", "note": "ghi chú"}
    return inf[[c for c in keep if c in inf.columns]].rename(columns=keep)


def sheet_summary(runs, inf, final_df, top: int = 10) -> pd.DataFrame:
    rows = []
    for _, r in runs.iterrows():
        rows.append({"nguồn": "train", "exp_id": f"{r.exp_id} (seed{r.seed})", "mô tả": r.backbone,
                     "macro-F1 val": r.val_macro_f1, "top-1 val": r.val_top1,
                     "p50 batch-1 (ms)": r.latency_b1_p50_ms, "tham số (M)": r.params_M})
    for _, r in inf.iterrows():
        rows.append({"nguồn": "suy luận", "exp_id": r.exp_id, "mô tả": r.method, "macro-F1 val": r.macro_f1_val,
                     "top-1 val": r.top1_val, "p50 batch-1 (ms)": r.get("p50_ms", np.nan), "tham số (M)": np.nan})
    df = pd.DataFrame(rows).sort_values("macro-F1 val", ascending=False).head(top)
    return df.reset_index(drop=True)


def build_xlsx(out_path, runs_dir, inf, lat, pred_dir, labels_dir, final_ids: dict[str, str]):
    """final_ids ví dụ {"T00": "nền + 1 view", "F01": "<backbone> + <công thức> + <suy luận>"}."""
    ev = import_eval()
    runs = collect_runs(runs_dir)
    t00 = runs[(runs.exp_id == "T00")]
    noise = float(t00.val_macro_f1.std(ddof=1)) if len(t00) >= 3 else None
    final_df, per_df = sheet_final(ev, pred_dir, labels_dir, final_ids)
    sheets = {
        "Backbones": sheet_backbones(runs), "Training": sheet_training(runs, noise),
        "Inference": sheet_inference(inf), "Final": final_df, "PerClass": per_df,
        "Latency": lat.rename(columns={"config": "cấu hình", "fused_bn": "gộp BN", "p50": "p50 (ms)",
                                       "p95": "p95 (ms)", "p99": "p99 (ms)", "img_s": "ảnh/s"}),
        "Summary": sheet_summary(runs, inf, final_df),
    }
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)
            ws = w.sheets[name]
            ws.freeze_panes = "A2"
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 48)
            for row in ws.iter_rows(min_row=2):
                for c in row:
                    if isinstance(c.value, float):
                        c.number_format = "0.0000"
    print(f"Đã ghi {out_path} | nhiễu seed (std macro-F1 val của T00) = {noise}")
    return sheets
