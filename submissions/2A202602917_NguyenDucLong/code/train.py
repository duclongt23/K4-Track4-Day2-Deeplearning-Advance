"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Một hàm `run(cfg)` dùng chung cho mọi cấu hình (RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0

Chỉ số chọn checkpoint (macro-F1 val) dùng `eval.compute_metrics` của repo gốc để cùng định nghĩa lúc chấm.
`eval.py` được tìm tự động ở các thư mục cha của file này (hoặc biến môi trường EVAL_DIR).
"""
from __future__ import annotations

import copy
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def import_eval():
    """Import eval.py của repo gốc: tìm ở $EVAL_DIR, rồi các thư mục cha của file này, rồi thư mục hiện tại."""
    cands = [Path(os.environ["EVAL_DIR"])] if os.environ.get("EVAL_DIR") else []
    cands += list(Path(__file__).resolve().parents) + [Path.cwd()] + list(Path.cwd().parents)
    for d in cands:
        if (d / "eval.py").exists():
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))
            import eval as ev
            return ev
    raise FileNotFoundError("Không tìm thấy eval.py; đặt biến môi trường EVAL_DIR trỏ tới thư mục chứa nó.")


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    curves_dir: str = "curves"        # ảnh đường cong: <curves_dir>/<exp_id>_<backbone>.png
    save_val_predictions: bool = False  # ghi predictions/<exp_id>_seed<k>_val.csv
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    # Với cấu hình chung kết nên dùng final.finalize() (một lượt test, có TTA/temperature scaling).
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định random, numpy, torch (CPU + CUDA). Worker DataLoader được seed trong dataset._seed_worker.

    cudnn.deterministic=True nhưng benchmark=False: tái lập tốt hơn, đổi lại chậm hơn một chút. Một số phép
    CUDA (atomicAdd trong backward) vẫn không tất định tuyệt đối; ghi điều này vào báo cáo.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg: Config):
    """AdamW với 3 nhóm tham số (xem model.param_groups): weight decay không áp dụng cho norm/bias."""
    from model import param_groups
    return torch.optim.AdamW(param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay))


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính (theo bước) rồi cosine về ~0 (slide trang 55)."""
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = max(1, int(cfg.warmup_epochs * steps_per_epoch))
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56).

    Giữ bản sao float của state_dict (gồm cả buffer BatchNorm: running_mean/var cũng được làm mượt).
    `copy_to(model)` chép trọng số EMA vào một model để đánh giá.
    """

    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model) -> None:
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v)


def _set_frozen_bn_eval(model) -> None:
    """Backbone đóng băng thì BatchNorm phải ở eval (không cập nhật thống kê); head vẫn train."""
    head_ids = {id(m) for m in model.get_classifier().modules()}
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and id(m) not in head_ids:
            m.eval()


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện. Trả về {"train_loss", "lr"}."""
    from losses import mix_batch, mixed_loss
    model.train()
    if cfg.init == "frozen":
        _set_frozen_bn_eval(model)

    total_loss, n_batches = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        targets = y
        if cfg.mix:
            x, targets = mix_batch(x, y, cfg.mix_alpha, cfg.mix)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
            logits = model(x)
        # loss tính ở fp32 cho ổn định (focal / label smoothing)
        loss = mixed_loss(criterion, logits.float(), targets) if cfg.mix else criterion(logits.float(), targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        n_batches += 1
    return {"train_loss": total_loss / max(1, n_batches), "lr": scheduler.get_last_lr()[0]}


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    """Chạy model ở chế độ eval, KHÔNG gradient. Trả về (filenames, y_true, logits, loss).

    `criterion=None` -> loss là cross-entropy thường, để val_loss so sánh được giữa các loại loss khác nhau.
    """
    model.eval()
    fnames, ys, outs, total, n = [], [], [], 0.0, 0
    for x, y, fn in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x).float()
        loss = criterion(logits, y) if criterion is not None else torch.nn.functional.cross_entropy(logits, y)
        total += loss.item() * len(y)
        n += len(y)
        fnames.extend(fn)
        ys.append(y.cpu().numpy())
        outs.append(logits.cpu().numpy())
    return fnames, np.concatenate(ys), np.concatenate(outs), total / max(1, n)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Đường cong training của một thí nghiệm -> curves/<exp_id>_<mota>.png (GUIDE.md mục 6.2).

    3 khung: loss train/val, macro-F1 + top-1 val, LR theo epoch (thấy warmup + cosine).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], "o-", label="train loss")
    ax[0].plot(ep, [h["val_loss"] for h in history], "s--", label="val loss (CE)")
    ax[0].set(xlabel="epoch", ylabel="loss", title="Loss")
    ax[1].plot(ep, [h["val_macro_f1"] for h in history], "o-", label="val macro-F1")
    ax[1].plot(ep, [h["val_top1"] for h in history], "s--", label="val top-1")
    best = max(history, key=lambda h: h["val_macro_f1"])
    ax[1].axvline(best["epoch"], color="gray", ls=":", label=f"best epoch {best['epoch']}")
    ax[1].set(xlabel="epoch", ylabel="metric (val)", title="Metric trên val")
    ax[2].plot(ep, [h["lr"] for h in history], "o-", label="LR (cuối epoch, nhóm 0)")
    ax[2].set(xlabel="epoch", ylabel="learning rate", title="LR")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _cfg_dict(cfg: Config) -> dict:
    return json.loads(json.dumps(asdict(cfg)))


def build_eval_loaders(cfg: Config, val_df, test_df=None):
    """DataLoader val (và test nếu có) với tiền xử lý đánh giá: không augmentation, thứ tự theo df."""
    import dataset
    tf = dataset.build_transforms(False, cfg.img_size, cfg.aug)
    val_loader = dataset.make_loader(val_df, cfg.images_dir, tf, cfg.batch_size, False, None, cfg.num_workers)
    test_loader = None
    if test_df is not None:
        test_loader = dataset.make_loader(test_df, cfg.images_dir, tf, cfg.batch_size, False, None, cfg.num_workers)
    return val_loader, test_loader


def run(cfg: Config, force: bool = False) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt.

    Lưu vào run_dir(cfg): config.json, history.csv, best.pt, val_logits.npy, result.json.
    Nếu result.json đã có và config.json giống hệt thì bỏ qua (tiếp tục được khi phiên Colab/Kaggle bị ngắt).
    """
    import dataset
    import losses
    from model import build_model, count_gmacs, count_params, weights_tag

    ev = import_eval()
    out_dir = run_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_json = _cfg_dict(cfg)
    if not force and (out_dir / "result.json").exists() and (out_dir / "config.json").exists():
        if json.loads((out_dir / "config.json").read_text()) == cfg_json:
            print(f"[skip] {cfg.exp_id} seed{cfg.seed}: đã có result.json")
            return json.loads((out_dir / "result.json").read_text())
    (out_dir / "config.json").write_text(json.dumps(cfg_json, indent=2))

    set_seed(cfg.seed)
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, cfg.fold)
    dataset.check_split(train_df, val_df, test_df, cfg.images_dir)

    train_tf = dataset.build_transforms(True, cfg.img_size, cfg.aug)
    train_loader = dataset.make_loader(train_df, cfg.images_dir, train_tf, cfg.batch_size, True,
                                       cfg.sampler, cfg.num_workers)
    val_loader, _ = build_eval_loaders(cfg, val_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.backbone, True, 9, cfg.drop_rate, cfg.init).to(device)
    tag = weights_tag(model)

    kw = {"smoothing": cfg.label_smoothing or 0.1, "gamma": cfg.focal_gamma}
    if cfg.loss == "ce_weighted":
        counts = train_df["Label"].value_counts().sort_index().values  # chỉ dùng số liệu TRAIN
        kw["weight"] = losses.class_weights(counts, cfg.class_weight_beta).to(device)
    criterion = losses.build_criterion(cfg.loss, **kw).to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None
    eval_model = copy.deepcopy(model) if ema else model  # EMA: đánh giá trên bản sao mang trọng số EMA

    best_f1, best_epoch, best_logits, history = -1.0, -1, None, []
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        res = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        t_train = time.time() - t0

        if ema:
            ema.copy_to(eval_model)
        fnames, y_true, logits, val_loss = evaluate(eval_model, val_loader, None, device)
        probs = softmax_np(logits)
        m = ev.compute_metrics(y_true, probs.argmax(1), probs)
        history.append({"epoch": epoch, "train_loss": res["train_loss"], "val_loss": val_loss,
                        "val_macro_f1": m["macro_f1"], "val_top1": m["top1"], "lr": res["lr"], "time": t_train})
        print(f"[{cfg.exp_id} s{cfg.seed}] ep {epoch}/{cfg.epochs} train {res['train_loss']:.4f} | "
              f"val {val_loss:.4f} | F1 {m['macro_f1']:.4f} | top1 {m['top1']:.4f} | {t_train:.0f}s")

        if m["macro_f1"] > best_f1:  # strict '>' -> hòa thì giữ epoch sớm hơn
            best_f1, best_epoch, best_logits = m["macro_f1"], epoch, logits
            torch.save(eval_model.state_dict(), out_dir / "best.pt")
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    np.save(out_dir / "val_logits.npy", best_logits)
    probs = softmax_np(best_logits)
    best_m = ev.compute_metrics(y_true, probs.argmax(1), probs)
    if cfg.save_val_predictions:
        ev.save_predictions(pred_path(cfg, "val"), fnames, y_true, probs)

    plot_curves(history, Path(cfg.curves_dir) / f"{cfg.exp_id}_{cfg.backbone}.png",
                f"{cfg.exp_id} | {cfg.backbone} | seed {cfg.seed}")

    # Độ trễ sơ bộ batch 1 (đo kỹ ở Bước 3 bằng benchmark.latency_report)
    lat = float("nan")
    try:
        from benchmark import latency_report
        lat = latency_report(copy.deepcopy(eval_model), 1, cfg.img_size, "fp32", device.type,
                             warmup=10, iters=50)["p50"]
    except Exception as e:
        print("latency bỏ qua:", e)

    result = {
        "exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": cfg.backbone, "weights_tag": tag,
        "params_M": count_params(eval_model), "gmacs": count_gmacs(eval_model, cfg.img_size),
        "img_size": cfg.img_size, "epochs": cfg.epochs, "best_epoch": best_epoch,
        "val_macro_f1": best_m["macro_f1"], "val_top1": best_m["top1"], "val_balanced_acc": best_m["balanced_acc"],
        "val_ece": best_m["ece"], "val_f1_per_class": [float(v) for v in best_m["f1"]],
        "val_recall_per_class": [float(v) for v in best_m["recall"]],
        "time_per_epoch_s": float(np.mean([h["time"] for h in history])),
        "latency_b1_p50_ms": lat, "device": device.type,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "torch": torch.__version__,
    }
    try:
        import timm
        result["timm"] = timm.__version__
    except ImportError:
        pass
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))

    if cfg.save_test_predictions:  # chỉ Bước 4; ưu tiên final.finalize() cho cấu hình chung kết
        _, test_loader = build_eval_loaders(cfg, val_df, test_df)
        tf, ty, tl, _ = evaluate(eval_model, test_loader, None, device)
        ev.save_predictions(pred_path(cfg, "test"), tf, ty, softmax_np(tl))
    return result


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict."""
    import ast
    d = {}
    for p in pairs:
        k, v = p.split("=", 1)
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            v = {"true": True, "false": False, "none": None}.get(v.lower(), v)
        d[k] = v
    return d


def main() -> None:
    """Điểm vào dòng lệnh: python train.py --set exp_id=B01 backbone=resnet50 seed=0"""
    args = sys.argv[1:]
    overrides = parse_overrides(args[args.index("--set") + 1:]) if "--set" in args else {}
    print("Run completed:", run(Config(**overrides)))


if __name__ == "__main__":
    main()
