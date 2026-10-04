"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Liên hệ slide Day 2: label smoothing (trang 56), focal loss (trang 57), Mixup/CutMix (trang 48).

Giao diện bạn phải giữ:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: "ce", "ls" (label smoothing), "focal", "ce_weighted".

    Ví dụ kw: smoothing=0.1, gamma=2.0, alpha=None, weight=tensor.
    tạo đúng loss, hoặc gọi các lớp bên dưới.
    """
    if kind == "ce":
        return nn.CrossEntropyLoss()
    elif kind == "ls":
        return LabelSmoothingCE(smoothing=kw.get("smoothing", 0.1))
    elif kind == "focal":
        return FocalLoss(gamma=kw.get("gamma", 2.0), alpha=kw.get("alpha", None))
    elif kind == "ce_weighted":
        return nn.CrossEntropyLoss(weight=kw.get("weight", None))
    else:
        raise ValueError(f"Unknown criterion kind: {kind}")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K  (slide trang 56).

    tự cài đặt hoặc dùng torch.nn.CrossEntropyLoss(label_smoothing=eps), rồi ghi rõ
    bạn đã chọn cách nào. Kiểm tra: eps = 0 phải cho đúng CE.
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
    def forward(self, logits, targets):
        return self.criterion(logits, targets)


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)  (slide trang 57).

    Cách làm:
      - tính log_softmax, lấy p_t của lớp đúng, nhân (1 - p_t)^gamma, lấy trung bình batch
      - alpha: None hoặc vector trọng số theo lớp
    BẮT BUỘC viết một kiểm tra nhỏ: gamma = 0 phải cho đúng cross-entropy (sai số < 1e-6).
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits.float(), targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        return focal_loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN.

    - beta = 0: trọng số tỉ lệ nghịch với số ảnh (1 / n_c), chuẩn hoá về trung bình 1
    - beta > 0: class-balanced theo "số mẫu hiệu dụng": w_c = (1 - beta) / (1 - beta ** n_c)
      (slide trang 57, Cui et al. arXiv:1901.05555); chuẩn hoá tổng trọng số về số lớp

    trả về tensor độ dài 9. Chỉ dùng số liệu của train, không dùng val hay test.
    """
    counts = np.asarray(counts, dtype=np.float64)
    if not beta:                      # None hoặc 0.0
        w = 1.0 / counts
        w = w / w.mean()              # trung bình 1
    else:
        w = (1.0 - beta) / (1.0 - np.power(beta, counts))
        w = w / w.sum() * len(counts)  # tổng bằng số lớp
    return torch.tensor(w, dtype=torch.float32)

def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn.

    - lam ~ Beta(alpha, alpha)
    - mode="mixup": x_mix = lam * x + (1 - lam) * x[perm]
    - mode="cutmix": cắt một hộp chữ nhật từ x[perm] dán vào x, rồi điều chỉnh lam theo
      DIỆN TÍCH THỰC của hộp sau khi cắt ra ngoài biên (slide trang 48)
    - trả về (x_mix, (y_a, y_b, lam)) với y_a = y, y_b = y[perm]

    tự cài đặt. Kiểm tra bằng mắt: vẽ vài ảnh sau khi trộn và in lam.
    """
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    
    if mode == "mixup":
        x_mixed = lam * x + (1 - lam) * x[index]
    elif mode == "cutmix":
        H, W = x.size(2), x.size(3)
        cut_rat = np.sqrt(1.0 - lam)
        cut_h, cut_w = int(H * cut_rat), int(W * cut_rat)

        cy, cx = np.random.randint(H), np.random.randint(W)
        y1, y2 = int(np.clip(cy - cut_h // 2, 0, H)), int(np.clip(cy + cut_h // 2, 0, H))
        x1, x2 = int(np.clip(cx - cut_w // 2, 0, W)), int(np.clip(cx + cut_w // 2, 0, W))

        x_mixed = x.clone()
        x_mixed[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
        # lam điều chỉnh theo DIỆN TÍCH THỰC của hộp sau khi cắt ra ngoài biên
        lam = 1.0 - ((y2 - y1) * (x2 - x1) / (H * W))
    else:
        x_mixed = x
        lam = 1.0
        
    y_a, y_b = y, y[index]
    return x_mixed, (y_a, y_b, lam)


def mixed_loss(criterion, logits, targets):
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b).

   . Lưu ý: accuracy trên batch đã trộn không còn nghĩa bình thường; đánh giá bằng val.
    """
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
