"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp dụng sang test (README.md, S2 và S4).
"""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.fusion import fuse_conv_bn_eval


def predict_logits(model, loader, device, view=None):
    """Chạy model trên loader và gom logit theo đúng thứ tự file.
    `view` là hàm biến đổi batch ảnh trước khi đưa vào model (ví dụ lật ngang), hoặc None.
    Trả về (filenames, y_true, logits[N, 9]) dạng numpy.
    """
    model.eval()
    filenames = []
    y_true_list = []
    logits_list = []

    with torch.inference_mode():
        for batch in loader:
            if isinstance(batch, dict):
                imgs = batch["image"]
                fns = batch.get("filename", None)
                labels = batch.get("label", None)
            else:
                imgs = batch[0]
                labels = batch[1] if len(batch) > 1 else None
                fns = batch[2] if len(batch) > 2 else None

            imgs = imgs.to(device)
            if view is not None:
                imgs = view(imgs)

            out = model(imgs)
            logits_list.append(out.detach().cpu().numpy())

            if labels is not None:
                if torch.is_tensor(labels):
                    y_true_list.append(labels.detach().cpu().numpy())
                else:
                    y_true_list.append(np.array(labels))
            if fns is not None:
                filenames.extend(fns if isinstance(fns, list) else list(fns))

    logits_arr = np.concatenate(logits_list, axis=0) if len(logits_list) > 0 else np.empty((0, 9))
    y_true_arr = np.concatenate(y_true_list, axis=0) if len(y_true_list) > 0 else None
    return filenames, y_true_arr, logits_arr


def view_identity(x):
    """Giữ nguyên batch ảnh."""
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W) bằng torch.flip theo chiều W (slide trang 75)."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x, crop: int):
    """5 crop (4 góc + giữa) kích thước `crop` cho batch x (N, C, H, W). Trả về list các batch."""
    H, W = x.shape[-2:]
    h_c, w_c = min(crop, H), min(crop, W)
    top_left = x[..., 0:h_c, 0:w_c]
    top_right = x[..., 0:h_c, W - w_c : W]
    bot_left = x[..., H - h_c : H, 0:w_c]
    bot_right = x[..., H - h_c : H, W - w_c : W]
    c_y, c_x = (H - h_c) // 2, (W - w_c) // 2
    center = x[..., c_y : c_y + h_c, c_x : c_x + w_c]
    return [top_left, top_right, bot_left, bot_right, center]


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes`, trả về list các batch."""
    views = []
    for s in sizes:
        resized = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
        views.append(resized)
    return views


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62).
      - space="prob":  trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    Trả về xác suất (N, 9) đã chuẩn hoá.
    """
    if space == "prob":
        probs = [F.softmax(torch.as_tensor(l), dim=-1).numpy() for l in logits_per_view]
        return np.mean(probs, axis=0)
    elif space == "logit":
        avg_logits = np.mean(logits_per_view, axis=0)
        return F.softmax(torch.as_tensor(avg_logits), dim=-1).numpy()
    else:
        raise ValueError(f"Unknown aggregation space: {space}")


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed)."""
    return np.mean(list_of_probs, axis=0)


def fit_temperature(val_logits, val_labels) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T)  (slide trang 69).
    Tối ưu hoá tham số log T bằng LBFGS trên cross entropy loss.
    """
    logits_t = torch.as_tensor(val_logits, dtype=torch.float32)
    labels_t = torch.as_tensor(val_labels, dtype=torch.long)
    log_T = nn.Parameter(torch.zeros(1, dtype=torch.float32))
    optimizer = optim.LBFGS([log_T], lr=0.05, max_iter=100)
    criterion = nn.CrossEntropyLoss()

    def eval_step():
        optimizer.zero_grad()
        T = torch.exp(log_T)
        loss = criterion(logits_t / T, labels_t)
        loss.backward()
        return loss

    optimizer.step(eval_step)
    T_opt = float(torch.exp(log_T).item())
    return max(T_opt, 1e-3)


def apply_temperature(logits, T: float):
    """Trả về xác suất softmax(logits / T)."""
    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    return F.softmax(logits_t / float(T), dim=-1).numpy()


def fuse_conv_bn(model, check_input_size: int = 224, tol: float = 1e-3):
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75).

    Duyệt từng module: nếu hai con liên tiếp là (Conv2d, BatchNorm2d) thì gộp. Với `BatchNormAct2d` của timm
    (BN + kích hoạt trong một module) thay BN bằng đúng lớp kích hoạt của nó, để không mất ReLU/SiLU.
    Sau khi gộp, so sánh logit với model gốc trên đầu vào ngẫu nhiên; nếu lệch quá `tol` thì trả lại model gốc.
    Kiến trúc không có BN (ViT, Swin, ConvNeXt dùng LayerNorm) cho kết quả không đổi (số cặp gộp = 0).
    Gán `model_fused.n_fused` = số cặp đã gộp.
    """
    fused = copy.deepcopy(model).cpu().eval()
    n = 0
    for parent in fused.modules():
        names = list(parent._modules.keys())
        for i in range(len(names) - 1):
            conv, bn = parent._modules[names[i]], parent._modules[names[i + 1]]
            if (isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d)
                    and bn.num_features == conv.out_channels and bn.track_running_stats):
                parent._modules[names[i]] = fuse_conv_bn_eval(conv, bn)
                parent._modules[names[i + 1]] = getattr(bn, "act", None) or nn.Identity()
                n += 1
    ref = copy.deepcopy(model).cpu().eval()
    x = torch.randn(2, 3, check_input_size, check_input_size)
    with torch.inference_mode():
        err = (ref(x) - fused(x)).abs().max().item()
    if err > tol:
        print(f"fuse_conv_bn: lệch {err:.2e} > {tol}, giữ model gốc")
        ref.n_fused = 0
        return ref
    fused.n_fused = n
    return fused
