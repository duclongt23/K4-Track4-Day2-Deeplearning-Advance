"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

PSEUDO-CODE: chỉ có khung (cấu hình và quy ước đặt tên file); bạn tự hoàn thiện mọi hàm có
`raise NotImplementedError` và các bước TODO trong `run()`. Dùng MỘT hàm `run(cfg)` cho mọi cấu hình
(RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số dùng để chọn checkpoint (macro-F1 val) phải tính bằng eval.compute_metrics của repo gốc,
để cùng định nghĩa với lúc chấm:
    sys.path.insert(0, "<thư mục chứa eval.py>");  from eval import compute_metrics
"""
import imp
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Ghi file dự đoán đúng định dạng bằng hàm có sẵn trong eval.py (repo gốc):
#     from eval import save_predictions, compute_metrics
# Log theo epoch (history.csv) và config.json bạn tự ghi bằng pandas/json.


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
    aug: str = "basic"                # basic | color | trivial | randaug ...
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
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"

import random, numpy as np, torch
def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên.

    TODO: random, numpy, torch (CPU và CUDA); cân nhắc cudnn.deterministic/benchmark và
    seed cho worker của DataLoader. Ghi lại trong báo cáo mức độ tái lập bạn đạt được.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def build_optimizer(model, cfg: Config):
    """AdamW với 3 nhóm tham số (xem model.param_groups). TODO."""
    from model import param_groups
    groups = param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về ~0 (slide trang 55). TODO."""
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)
    
    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=max(1, warmup_steps))
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56).

    TODO:
      - __init__(model, decay): sao chép trọng số
      - update(model): sau mỗi bước tối ưu
      - copy_to(model) hoặc dùng bản sao riêng để đánh giá bằng trọng số EMA
      - lưu ý BatchNorm: buffer (running_mean/var) cũng phải được xử lý hợp lý
    """

    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {}
        for name, param in model.state_dict().items():
            if param.dtype.is_floating_point:
                self.shadow[name] = param.clone().detach()

    def update(self, model) -> None:
        for name, param in model.state_dict().items():
            if name in self.shadow and param.dtype.is_floating_point:
                self.shadow[name].sub_((1.0 - self.decay) * (self.shadow[name] - param.detach()))
                
    def copy_to(self, model):
        for name, param in model.state_dict().items():
            if name in self.shadow:
                param.copy_(self.shadow[name])


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện. Trả về dict, ví dụ {"train_loss": ..., "lr": ...}."""
    model.train()
    if cfg.init == "frozen":
        from model import freeze_backbone
        freeze_backbone(model)
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
                
    total_loss = 0.0
    from losses import mix_batch, mixed_loss
    
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        
        if cfg.mix:
            x, targets = mix_batch(x, y, cfg.mix_alpha, cfg.mix)
        else:
            targets = y
            
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', enabled=cfg.amp):
            logits = model(x)
            if cfg.mix:
                loss = mixed_loss(criterion, logits, targets)
            else:
                loss = criterion(logits, targets)
                
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        if ema:
            ema.update(model)
            
        total_loss += loss.item()
        
    return {"train_loss": total_loss / len(loader), "lr": scheduler.get_last_lr()[0]}


def evaluate(model, loader, criterion, device):
    """Chạy model trên một loader ở chế độ eval, KHÔNG tính gradient."""
    model.eval()
    all_filenames = []
    all_y_true = []
    all_logits = []
    total_loss = 0.0
    
    with torch.inference_mode():
        for x, y, fnames in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            
            all_filenames.extend(fnames)
            all_y_true.append(y.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            
    import numpy as np
    y_true = np.concatenate(all_y_true)
    logits = np.concatenate(all_logits)
    loss = total_loss / len(loader)
    
    return all_filenames, y_true, logits, loss


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Vẽ đường cong training của một thí nghiệm -> curves/<exp_id>_<mota>.png (GUIDE.md mục 6.2)."""
    import matplotlib.pyplot as plt
    epochs = [h["epoch"] for h in history]
    train_loss = [h.get("train_loss", 0) for h in history]
    val_loss = [h.get("val_loss", 0) for h in history]
    val_f1 = [h.get("val_macro_f1", 0) for h in history]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(epochs, train_loss, color='tab:red', linestyle='-', label='Train Loss')
    ax1.plot(epochs, val_loss, color='tab:red', linestyle='--', label='Val Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Macro F1', color='tab:blue')
    ax2.plot(epochs, val_f1, color='tab:blue', linestyle='-', label='Val F1')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    fig.tight_layout()
    plt.title(title)
    fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close(fig)


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt."""
    import json, time, sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from eval import compute_metrics, save_predictions
    import dataset, losses
    from model import build_model, count_params, count_gmacs
    
    set_seed(cfg.seed)
    out_dir = run_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg.__dict__, f, indent=4)
        
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, cfg.fold)
    dataset.check_split(train_df, val_df, test_df, cfg.images_dir)
    
    train_tf = dataset.build_transforms(True, cfg.img_size, cfg.aug)
    val_tf = dataset.build_transforms(False, cfg.img_size, cfg.aug)
    
    train_loader = dataset.make_loader(train_df, cfg.images_dir, train_tf, cfg.batch_size, True, cfg.sampler, cfg.num_workers)
    val_loader = dataset.make_loader(val_df, cfg.images_dir, val_tf, cfg.batch_size, False, None, cfg.num_workers)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.backbone, True, 9, cfg.drop_rate, cfg.init).to(device)
    
    kw = {"smoothing": cfg.label_smoothing, "gamma": cfg.focal_gamma}
    if cfg.loss == "ce_weighted":
        kw["weight"] = losses.class_weights(train_df["Label"].value_counts().sort_index().values, cfg.class_weight_beta).to(device)
    criterion = losses.build_criterion(cfg.loss, **kw).to(device)
    
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None
    
    best_f1 = -1.0
    best_epoch = -1
    history = []
    
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_res = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        t_train = time.time() - t0
        
        eval_model = model
        if ema:
            eval_model = build_model(cfg.backbone, False, 9, 0.0, "scratch").to(device)
            ema.copy_to(eval_model)
            
        fnames, y_true, logits, val_loss = evaluate(eval_model, val_loader, criterion, device)
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        metrics = compute_metrics(y_true, probs.argmax(axis=1), probs)
        
        val_f1 = metrics["macro_f1"]
        print(f"Epoch {epoch}: Train Loss {train_res['train_loss']:.4f} | Val Loss {val_loss:.4f} | Val F1 {val_f1:.4f} | Time {t_train:.1f}s")
        
        history.append({
            "epoch": epoch,
            "train_loss": train_res["train_loss"],
            "val_loss": val_loss,
            "val_macro_f1": val_f1,
            "time": t_train,
            "lr": train_res["lr"]
        })
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            torch.save(eval_model.state_dict(), out_dir / "best.pt")
            
    import pandas as pd
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    
    eval_model = build_model(cfg.backbone, False, 9, 0.0, "scratch").to(device)
    eval_model.load_state_dict(torch.load(out_dir / "best.pt"))
    
    Path(cfg.pred_dir).mkdir(exist_ok=True)
    
    val_fnames, val_y, val_logits, _ = evaluate(eval_model, val_loader, criterion, device)
    val_probs = torch.softmax(torch.tensor(val_logits), dim=1).numpy()
    save_predictions(val_fnames, val_y, val_probs.argmax(axis=1), val_probs, pred_path(cfg, "val"))
    
    if cfg.save_test_predictions:
        test_tf = dataset.build_transforms(False, cfg.img_size, cfg.aug)
        test_loader = dataset.make_loader(test_df, cfg.images_dir, test_tf, cfg.batch_size, False, None, cfg.num_workers)
        test_fnames, test_y, test_logits, _ = evaluate(eval_model, test_loader, criterion, device)
        test_probs = torch.softmax(torch.tensor(test_logits), dim=1).numpy()
        save_predictions(test_fnames, test_y, test_probs.argmax(axis=1), test_probs, pred_path(cfg, "test"))
        
    plot_curves(history, Path("curves") / f"{cfg.exp_id}_{cfg.backbone}.png", f"{cfg.exp_id} - {cfg.backbone}")
    
    params = count_params(eval_model)
    gmacs = count_gmacs(eval_model, cfg.img_size)
    
    return {
        "best_epoch": best_epoch,
        "val_macro_f1": best_f1,
        "time_per_epoch": sum(h["time"] for h in history) / len(history),
        "params_M": params,
        "gmacs": gmacs
    }


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict."""
    import ast
    d = {}
    for p in pairs:
        k, v = p.split("=")
        try:
            v = ast.literal_eval(v)
        except:
            if v.lower() == "true": v = True
            elif v.lower() == "false": v = False
            elif v.lower() == "none": v = None
        d[k] = v
    return d


def main() -> None:
    """Điểm vào dòng lệnh."""
    import sys
    args = sys.argv[1:]
    overrides = {}
    if "--set" in args:
        idx = args.index("--set")
        overrides = parse_overrides(args[idx+1:])
    cfg = Config(**overrides)
    res = run(cfg)
    print("Run completed:", res)


if __name__ == "__main__":
    main()
