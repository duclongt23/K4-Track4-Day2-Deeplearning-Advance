"""experiments.py - bảng thí nghiệm của bài lab (một nơi duy nhất; notebook chỉ gọi các hàm ở đây).

Mọi thí nghiệm khác `T00` đúng MỘT yếu tố (nguyên tắc N1), trừ các dòng kết hợp được đánh dấu rõ.
`common` là dict dùng chung (đường dẫn, epochs, batch size...) để mọi cấu hình cùng công thức nền.
"""
from __future__ import annotations

from dataclasses import replace

from train import Config

# (exp_id, tên timm, ghi chú) - đủ ràng buộc RUBRIC B: ResNet, ResNeXt/ConvNeXt, transformer, mạng nhẹ
BACKBONES = [
    ("B01", "resnet50", "ResNet - mốc"),
    ("B02", "resnext50_32x4d", "ResNeXt"),
    ("B03", "convnext_tiny", "ConvNeXt - ResNet hiện đại hoá"),
    ("B04", "deit_small_patch16_224", "Transformer (DeiT-S)"),
    ("B05", "efficientnet_b0", "Mạng nhẹ"),
    ("B06", "mobilenetv3_large_100", "Mạng nhẹ nhất"),
]


def backbone_cfgs(common: dict, seed: int = 0) -> list[Config]:
    """Bước 1: cùng công thức nền, cùng seed, mỗi backbone một lần."""
    return [Config(exp_id=eid, backbone=name, seed=seed, **common) for eid, name, _ in BACKBONES]


# exp_id -> (trục, khác T00 ở điểm nào, kwargs đổi so với T00). Trục theo GUIDE mục 3.
TRAIN_AXES = {
    "T01": ("A. Khởi tạo", "đóng băng backbone, chỉ train head", dict(init="frozen")),
    "T02": ("A. Khởi tạo", "huấn luyện từ đầu (không tiền huấn luyện)", dict(init="scratch")),
    "T03": ("B. Augmentation", "+ ColorJitter", dict(aug="color")),
    "T04": ("B. Augmentation", "TrivialAugmentWide", dict(aug="trivial")),
    "T05": ("B. Augmentation", "Mixup (alpha=1)", dict(mix="mixup", mix_alpha=1.0)),
    "T06": ("B. Augmentation", "CutMix (alpha=1)", dict(mix="cutmix", mix_alpha=1.0)),
    "T07": ("C. Loss", "label smoothing 0.1", dict(loss="ls", label_smoothing=0.1)),
    "T08": ("C. Loss", "focal loss gamma=2", dict(loss="focal", focal_gamma=2.0)),
    "T09": ("C. Loss", "CE trọng số lớp (1/n_c)", dict(loss="ce_weighted", class_weight_beta=None)),
    "T10": ("D. Cân bằng mẫu", "sampler cân bằng lớp", dict(sampler="balanced")),
    "T11": ("E. LR", "cùng LR 1e-4 cho backbone và head", dict(lr_head=1e-4)),
    "T12": ("F. Chính quy hoá", "EMA trọng số (decay 0.998)", dict(ema_decay=0.998)),
}


def training_cfgs(backbone: str, common: dict, seed: int = 0) -> list[tuple[Config, str, str]]:
    """Bước 2: T00 + các biến thể trên MỘT backbone. Trả về [(cfg, trục, mô tả)]."""
    base = Config(exp_id="T00", backbone=backbone, seed=seed, **common)
    out = [(base, "nền", "công thức nền (GUIDE 1.4)")]
    for eid, (axis, desc, kw) in TRAIN_AXES.items():
        out.append((replace(base, exp_id=eid, **kw), axis, desc))
    return out


def combo_cfg(backbone: str, common: dict, winners: list[str], seed: int = 0, exp_id: str = "T13") -> Config:
    """Kết hợp các yếu tố thắng rõ ở Bước 2 (ví dụ winners=["T06", "T07", "T12"]). Đánh dấu là KẾT HỢP trong bảng."""
    cfg = Config(exp_id=exp_id, backbone=backbone, seed=seed, **common)
    for w in winners:
        cfg = replace(cfg, **TRAIN_AXES[w][2])
    return cfg
