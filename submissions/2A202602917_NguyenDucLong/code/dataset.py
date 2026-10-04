"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1. Đọc trước khi viết.

Giao diện bạn phải giữ (để notebook, train.py và eval.py ghép được với nhau):
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # đổi nếu trọng số timm bạn dùng yêu cầu mean/std khác
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame.
    KHÔNG sửa, lọc hay chia lại dữ liệu.

    Cách làm:
      - đọc ba file CSV bằng pandas
      - trả về (train_df, val_df, test_df)
    """
    labels_dir = Path(labels_dir)
    train_df = pd.read_csv(labels_dir / f"train_subset{fold}.csv")
    val_df = pd.read_csv(labels_dir / f"val_subset{fold}.csv")
    test_df = pd.read_csv(labels_dir / f"test_subset{fold}.csv")
    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    Kiểm tra, mỗi ý lỗi thì `assert` / raise để dừng ngay:
      1. số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20)
      2. giao của từng cặp tập theo Filename phải RỖNG (train∩val, train∩test, val∩test)
      3. hợp ba tập phải bằng đúng 17.509 ảnh
      4. mọi Filename đều tồn tại trong `images_dir`
    Trả về dict, ví dụ {"n": {...}, "per_class": {...}, "overlap": {...}} để dán vào báo cáo.
    """
    n = {
        "train": len(train_df), "val": len(val_df), "test": len(test_df),
        "total": len(train_df) + len(val_df) + len(test_df),
    }
    assert n["total"] == 17509, f"Tổng số ảnh phải là 17509, nhận {n['total']}"
    for k in ("train", "val", "test"):
        ratio = n[k] / n["total"]
        expected = {"train": 0.6, "val": 0.2, "test": 0.2}[k]
        assert abs(ratio - expected) < 0.01, f"Tỉ lệ {k} = {ratio:.3f}, lệch quá 1 điểm % so với {expected}"

    files = {"train": set(train_df["Filename"]), "val": set(val_df["Filename"]), "test": set(test_df["Filename"])}
    overlap = {
        "train&val": len(files["train"] & files["val"]),
        "train&test": len(files["train"] & files["test"]),
        "val&test": len(files["val"] & files["test"]),
    }
    assert all(v == 0 for v in overlap.values()), f"Giao các tập phải rỗng: {overlap}"
    assert len(files["train"] | files["val"] | files["test"]) == 17509, "Hợp ba tập phải đúng 17509 ảnh"

    per_class = {
        name: df["Label"].value_counts().sort_index().to_dict()
        for name, df in (("train", train_df), ("val", val_df), ("test", test_df))
    }

    images_dir = Path(images_dir)
    missing = [f for f in sorted(files["train"] | files["val"] | files["test"]) if not (images_dir / f).exists()]
    assert not missing, f"{len(missing)} file không có trong {images_dir}, ví dụ {missing[:3]}"

    print(f"Split OK | train {n['train']} val {n['val']} test {n['test']} (tổng {n['total']}) | giao rỗng | đủ file ảnh")
    return {"n": n, "per_class": per_class, "overlap": overlap, "missing_files": 0}


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo transform. `aug` chọn mức augmentation; bạn tự định nghĩa các giá trị.

    Gợi ý các giá trị `aug` (trục B của GUIDE.md mục 3): "basic", "color", "trivial", "randaug".
    Mixup/CutMix trộn theo batch nên nằm ở losses.py, không ở đây.

    Train (basic): RandomResizedCrop(img_size) + lật ngang + ToTensor + Normalize.
    Val/test: ảnh gốc 256x256 -> CenterCrop(img_size) (hoặc giữ nguyên 256; ghi rõ bạn chọn gì)
              + ToTensor + Normalize. KHÔNG augmentation ngẫu nhiên khi đánh giá.

    dùng torchvision.transforms (hoặc v2). Lưu ý: lật dọc có hợp lệ với ảnh cỏ dại không?
    """
    from torchvision import transforms as T

    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if not train:
        # Ảnh gốc 256x256. img_size <= 256: giữ 256 rồi CenterCrop(img_size); img_size > 256 (dò độ phân giải,
        # trục I04): phóng lên img_size, không crop. KHÔNG augmentation ngẫu nhiên khi đánh giá.
        return T.Compose([T.Resize(max(256, img_size)), T.CenterCrop(img_size), T.ToTensor(), normalize])

    # Chỉ lật ngang. Lật dọc/xoay chưa được kiểm chứng cho ảnh cỏ dại, nên không đưa vào nền (có thể thử riêng).
    geo = [T.RandomResizedCrop(img_size), T.RandomHorizontalFlip()]
    if aug == "basic":
        extra = []
    elif aug == "color":
        extra = [T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)]
    elif aug == "trivial":
        extra = [T.TrivialAugmentWide()]
    elif aug == "randaug":
        extra = [T.RandAugment(num_ops=2, magnitude=9)]
    else:
        raise ValueError(f"aug không hợp lệ: {aug!r} (basic | color | trivial | randaug)")
    return T.Compose(geo + extra + [T.ToTensor(), normalize])


class DeepWeedsDataset(torch.utils.data.Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) phải trả về (ảnh đã transform, nhãn int, tên file str).
    Tên file cần có để ghi `predictions/*.csv` đúng định dạng của eval.py.

    Cách làm:
      - __init__(self, df, images_dir, transform): giữ df, mở ảnh bằng PIL, chuyển sang RGB
      - __len__
      - __getitem__ -> (tensor, int(label), filename)
      - (tuỳ chọn) nạp trước ảnh vào RAM nếu bị nghẽn đọc đĩa trên Colab
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        filename = row["Filename"]
        label = int(row["Label"])
        
        img_path = self.images_dir / filename
        img = Image.open(img_path).convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, label, filename


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    """Tạo DataLoader.

    Cách làm:
      - train=True: shuffle (hoặc dùng sampler); train=False: không shuffle, giữ thứ tự df
        (thứ tự phải ổn định để ghép logit với Filename)
      - sampler=None | "balanced": "balanced" dùng WeightedRandomSampler với trọng số
        1/(số ảnh của lớp) (trục D của GUIDE.md mục 3)
      - drop_last=True khi train nếu batch cuối quá nhỏ làm BatchNorm không ổn định
      - pin_memory=True, num_workers hợp lý; seed cho worker (worker_init_fn) để tái lập
    """
    from torch.utils.data import DataLoader, WeightedRandomSampler

    dataset = DeepWeedsDataset(df, images_dir, transform=transform)
    pin = torch.cuda.is_available()
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin, drop_last=train,
                  worker_init_fn=_seed_worker, persistent_workers=num_workers > 0)

    if train and sampler == "balanced":
        counts = df["Label"].value_counts().sort_index()
        w = 1.0 / counts
        sample_weights = df["Label"].map(w).to_numpy()
        return DataLoader(dataset, sampler=WeightedRandomSampler(sample_weights, len(sample_weights)), **common)
    if sampler not in (None, "balanced"):
        raise ValueError(f"sampler không hợp lệ: {sampler!r}")
    return DataLoader(dataset, shuffle=train, **common)


def _seed_worker(worker_id: int) -> None:
    """Seed cho worker của DataLoader (numpy/random) từ seed torch của tiến trình chính."""
    import random
    import numpy as np
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
