from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets, transforms

VAL_SIZE = 1000
TEST_SIZE = 1000
TRAIN_SIZE = 1000
MNIST_RAW_FILES = (
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
)


def _mnist_cache_complete(raw: Path) -> bool:
    expected = {
        "train-images-idx3-ubyte": 47040016,
        "train-labels-idx1-ubyte": 60008,
        "t10k-images-idx3-ubyte": 7840016,
        "t10k-labels-idx1-ubyte": 10008,
    }
    return all(
        (raw / name).is_file() and (raw / name).stat().st_size == size
        for name, size in expected.items()
    )


def _download_mnist_to_cache(cache_root: Path) -> Path:
    cache_raw = cache_root / "MNIST" / "raw"
    cache_raw.mkdir(parents=True, exist_ok=True)
    datasets.MNIST(root=str(cache_root), train=True, download=True)
    datasets.MNIST(root=str(cache_root), train=False, download=True)
    if not _mnist_cache_complete(cache_raw):
        raise RuntimeError(f"bad mnist cache: {cache_raw}")
    return cache_raw


def _ensure_mnist_cache(data_dir: Path) -> Path:
    # demo reads via memmap; prefer ~/.cache over project data/ (iCloud timeouts)
    cache_root = Path.home() / ".cache" / "vit-ai"
    cache_raw = cache_root / "MNIST" / "raw"
    if _mnist_cache_complete(cache_raw):
        return cache_raw

    src_raw = Path(data_dir) / "MNIST" / "raw"
    if _mnist_cache_complete(src_raw):
        cache_raw.mkdir(parents=True, exist_ok=True)
        for name in MNIST_RAW_FILES:
            shutil.copy2(src_raw / name, cache_raw / name)
        return cache_raw

    for name in MNIST_RAW_FILES:
        partial = cache_raw / name
        if partial.is_file():
            partial.unlink()
    return _download_mnist_to_cache(cache_root)


def _open_idx_images(path: Path) -> np.memmap:
    with open(path, "rb") as f:
        _magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
    return np.memmap(path, dtype=np.uint8, mode="r", offset=16, shape=(n, rows, cols))


def _open_idx_labels(path: Path) -> np.memmap:
    with open(path, "rb") as f:
        _magic, n = struct.unpack(">II", f.read(8))
    return np.memmap(path, dtype=np.uint8, mode="r", offset=8, shape=(n,))


class MemmapConcatMNIST(Dataset[tuple[Tensor, int]]):
    N_TRAIN = 60000  # idx 0..59999 train, 60000..69999 test (matches ConcatDataset order)

    def __init__(self, data_dir: Path) -> None:
        raw = _ensure_mnist_cache(data_dir)
        self.train_images = _open_idx_images(raw / "train-images-idx3-ubyte")
        self.train_labels = _open_idx_labels(raw / "train-labels-idx1-ubyte")
        self.test_images = _open_idx_images(raw / "t10k-images-idx3-ubyte")
        self.test_labels = _open_idx_labels(raw / "t10k-labels-idx1-ubyte")

    def __len__(self) -> int:
        return self.N_TRAIN + len(self.test_images)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        if idx < self.N_TRAIN:
            img = np.array(self.train_images[idx], copy=True)
            label = int(self.train_labels[idx])
        else:
            j = idx - self.N_TRAIN
            img = np.array(self.test_images[j], copy=True)
            label = int(self.test_labels[j])
        x = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        return x, label


def build_memmap_mnist(data_dir: Path) -> MemmapConcatMNIST:
    return MemmapConcatMNIST(data_dir)


def build_full_mnist(data_dir: Path) -> Dataset[tuple[Tensor, int]]:
    tfm = transforms.ToTensor()
    train = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=tfm)
    test = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=tfm)
    return ConcatDataset([train, test])


def split_indices(
    n: int, *, val_size: int = VAL_SIZE, test_size: int = TEST_SIZE, seed: int
) -> tuple[list[int], list[int], list[int]]:
    if val_size <= 0 or test_size <= 0:
        raise ValueError("val_size and test_size must be positive.")
    if val_size + test_size >= n:
        raise ValueError("val_size + test_size must be smaller than dataset size.")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    test_idx = perm[:test_size]
    val_idx = perm[test_size : test_size + val_size]
    train_idx = perm[test_size + val_size :]  # remainder
    return (train_idx, val_idx, test_idx)


def load_test_indices(out_dir: Path) -> list[int]:
    return json.loads((out_dir / "test_indices.json").read_text())


def save_test_indices(out_dir: Path, test_idx: list[int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_indices.json").write_text(json.dumps(test_idx) + "\n")


def save_train_indices(out_dir: Path, train_idx: list[int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_indices.json").write_text(json.dumps(train_idx) + "\n")


def load_train_indices(out_dir: Path) -> list[int] | None:
    path = out_dir / "train_indices.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def subsample_indices(indices: list[int], size: int, seed: int) -> list[int]:
    if size <= 0 or size >= len(indices):
        return list(indices)
    g = torch.Generator().manual_seed(seed)
    pick = torch.randperm(len(indices), generator=g)[:size].tolist()
    return [indices[i] for i in pick]


def dataloader_kwargs(device: torch.device) -> dict:
    return {
        "num_workers": 0,  # fine for MNIST; avoids macOS spawn issues
        "pin_memory": device.type == "cuda",
    }


def make_loader(
    dataset: Dataset,
    indices: list[int],
    *,
    batch_size: int,
    device: torch.device,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        **dataloader_kwargs(device),
    )


@torch.inference_mode()
def recon_mse(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, count = (0.0, 0)
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        recon = model(x)
        total += float(torch.mean((recon - x) ** 2).item()) * x.size(0)
        count += int(x.size(0))
    return total / max(1, count)
