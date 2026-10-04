"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.


Giao diện bạn phải giữ:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

import copy

import timm
import torch

# Gợi ý backbone (GUIDE.md mục 2.1). Tag trọng số của timm có thể đổi theo phiên bản:
# dùng timm.list_pretrained("resnet50*") để xem, và GHI LẠI tag bạn dùng trong results.xlsx.
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      # hoặc vit_small_patch16_224
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Tạo model phân loại 9 lớp.

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ

    Cách làm:
      - timm.create_model(name, pretrained=..., num_classes=num_classes, drop_rate=...)
        (timm tự thay head mới; head khởi tạo ngẫu nhiên)
      - nếu init == "frozen": gọi freeze_backbone(model)
      - ghi lại tên tag trọng số thực sự được tải (model.pretrained_cfg)
    """
    if init not in ("scratch", "frozen", "finetune"):
        raise ValueError(f"init phải là scratch | frozen | finetune, nhận {init!r}")
    model = timm.create_model(name, pretrained=(init != "scratch"), num_classes=num_classes, drop_rate=drop_rate)
    if init == "frozen":
        freeze_backbone(model)
    return model


def weights_tag(model) -> str:
    """Tag trọng số timm thực sự được tải (ghi vào results.xlsx, cột `tag trọng số`)."""
    cfg = getattr(model, "pretrained_cfg", None) or {}
    return str(cfg.get("tag") or cfg.get("hf_hub_id") or cfg.get("url") or "none")


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head.

    Cách làm:
      - requires_grad = False cho tham số backbone; head (model.get_classifier()) vẫn train
      - lưu ý (GUIDE.md mục 3.2): backbone đóng băng thì BatchNorm cũng phải ở chế độ eval.
        Hãy nghĩ nơi nào trong train loop phải gọi lại model.train() mà vẫn giữ BN ở eval.
    """
    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_classifier().parameters():
        param.requires_grad = True


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52.

    - backbone có ndim > 1: lr = lr_backbone, weight_decay = weight_decay
    - norm và bias của backbone (ndim <= 1): lr = lr_backbone, weight_decay = 0
    - head mới: lr = lr_head (thường gấp 10 lần backbone), weight_decay = weight_decay

    Cách làm:
      - bỏ qua tham số requires_grad == False
      - trả về list[dict] dạng {"params": [...], "lr": ..., "weight_decay": ...}
      - (trục E) mở rộng: LR theo tầng nếu bạn muốn thử
    """
    groups = []
    backbone_params = []
    backbone_norm_bias = []

    classifier_ids = {id(p) for p in model.get_classifier().parameters()}

    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in classifier_ids:
            continue
        if p.ndim <= 1:
            backbone_norm_bias.append(p)
        else:
            backbone_params.append(p)
            
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay})
    if backbone_norm_bias:
        groups.append({"params": backbone_norm_bias, "lr": lr_backbone, "weight_decay": 0.0})
        
    head_params = [p for p in model.get_classifier().parameters() if p.requires_grad]
    if head_params:
        groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay})
        
    return groups


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng.."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size (MAC, không phải FLOPs 2x), đếm bằng `thop`.

    Chạy trên bản sao ở CPU để không làm bẩn model gốc (thop gắn thêm buffer vào module).
    Trả về nan nếu không có thop (cài bằng `pip install thop`).
    """
    try:
        from thop import profile
    except ImportError:
        return float("nan")
    m = copy.deepcopy(model).cpu().eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        macs, _ = profile(m, inputs=(dummy,), verbose=False)
    return macs / 1e9
