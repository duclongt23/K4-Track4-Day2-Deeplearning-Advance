"""auto_pipeline.py - chạy tự động Bước 2 -> 4 với quy tắc chọn CỐ ĐỊNH, chỉ dựa trên VAL.

Quy tắc (khai báo trước khi chạy, ghi vào báo cáo; mọi quyết định lưu ở <workdir>/auto_decisions.json):
  R1. NOISE = std (ddof=1) của macro-F1 val của T00 qua 3 seed.
  R2. Một yếu tố "thắng" nếu macro-F1 val tăng so với T00 hơn NOISE. Mỗi trục chỉ lấy yếu tố thắng tốt nhất.
  R3. T13 = kết hợp các yếu tố thắng. Cấu hình tốt nhất = T13 hoặc yếu tố đơn tốt nhất nếu hơn T00 quá NOISE,
      ngược lại giữ T00 (khi đó "không phân biệt được" với nền).
  R4. TTA lật ngang chỉ chọn nếu macro-F1 val tăng hơn NOISE so với I00. Temperature scaling chọn nếu ECE val
      (cross-fit) nhỏ hơn ECE của I00.
  R5. Test chạy đúng một lần mỗi seed qua final.finalize (có chốt chặn). Chạy lại sau khi bị ngắt thì bỏ qua seed đã xong.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import build_results
import experiments
import final
import inference_suite
from train import Config, run, run_dir


def run_all(common: dict, backbone: str, second_backbone: str, workdir: str, seeds=(0, 1, 2),
            keep: set[str] | None = None, p95_budget_ms: float = 100.0) -> dict:
    out_runs = common["out_dir"]
    decisions: dict = {"backbone": backbone, "second_backbone": second_backbone, "seeds": list(seeds)}

    # --- Bước 2: ablation (1 seed) + nhiễu qua 3 seed của T00 ---
    t_cfgs = experiments.training_cfgs(backbone, common, seed=0)
    for cfg, _, _ in t_cfgs:
        if keep is None or cfg.exp_id in keep or cfg.exp_id == "T00":
            run(cfg)
    t00 = t_cfgs[0][0]
    for s in seeds[1:]:
        run(replace(t00, seed=s))
    runs = build_results.collect_runs(out_runs)
    f1 = runs[runs.seed == 0].set_index("exp_id").val_macro_f1
    noise = float(runs[runs.exp_id == "T00"].val_macro_f1.std(ddof=1))
    decisions["noise"] = noise
    delta = {e: float(f1[e] - f1["T00"]) for e in f1.index if e.startswith("T") and e != "T00"}
    decisions["delta_vs_T00"] = delta

    best_per_axis: dict[str, str] = {}
    for e, d in delta.items():
        if e in experiments.TRAIN_AXES and d > noise:
            axis = experiments.TRAIN_AXES[e][0]
            if axis not in best_per_axis or d > delta[best_per_axis[axis]]:
                best_per_axis[axis] = e
    winners = sorted(best_per_axis.values())
    decisions["winners"] = winners

    candidates = {"T00": t00}
    if winners:
        combo = experiments.combo_cfg(backbone, common, winners, seed=0, exp_id="T13")
        run(combo)
        runs = build_results.collect_runs(out_runs)
        f1 = runs[runs.seed == 0].set_index("exp_id").val_macro_f1
        candidates["T13"] = combo
        for e in winners:
            candidates[e] = next(c for c, _, _ in t_cfgs if c.exp_id == e)
    ok = {e: c for e, c in candidates.items() if e == "T00" or f1[e] - f1["T00"] > noise}
    best_id = max(ok, key=lambda e: f1[e])
    best = ok[best_id]
    decisions["best_train_cfg"] = best_id

    # --- Bước 3: suy luận trên val ---
    second = next(c for c in experiments.backbone_cfgs(common) if c.backbone == second_backbone)
    inf_df, lat_df = inference_suite.run_suite(best, ensemble_cfgs=[second] if second.backbone != best.backbone else None)
    inf_df.to_csv(f"{workdir}/inference_val.csv", index=False)
    lat_df.to_csv(f"{workdir}/latency.csv", index=False)
    i = inf_df.set_index("exp_id")
    flip_gain = max(i.loc["I01", "macro_f1_val"], i.loc["I03", "macro_f1_val"]) - i.loc["I00", "macro_f1_val"]
    final_tta = "hflip" if flip_gain > noise else "none"
    final_ts = bool(i.loc["I07", "ece_val_crossfit"] < i.loc["I00", "ece_val"])
    b1 = lat_df[(lat_df.batch == 1) & lat_df.config.str.startswith("I00")]
    p95 = float(b1.p95.iloc[0]) if len(b1) else None
    decisions.update({"final_tta": final_tta, "final_ts": final_ts, "hflip_gain_val": float(flip_gain),
                      "p95_realtime_ms": p95, "p95_within_budget": bool(p95 is not None and p95 <= p95_budget_ms)})
    Path(workdir, "auto_decisions.json").write_text(json.dumps(decisions, indent=2))
    print("QUYẾT ĐỊNH:", json.dumps(decisions, indent=2))

    # --- Bước 4: chung kết + mốc, test một lần mỗi seed ---
    f01 = replace(best, exp_id="F01")
    for s in seeds:
        for cfg, tta, ts in ((replace(f01, seed=s), final_tta, final_ts), (replace(t00, seed=s), "none", False)):
            run(cfg)
            if (run_dir(cfg) / "test_done.json").exists():
                print(f"[skip finalize] {cfg.exp_id} seed{s}: test đã chạy")
                continue
            final.finalize(cfg, tta=tta, temperature=ts)
    return {"BEST_CFG": best, "F01": f01, "T00": t00, "FINAL_TTA": final_tta, "FINAL_TS": final_ts,
            "P95_REALTIME": p95, "inf_df": inf_df, "lat_df": lat_df, "decisions": decisions}
