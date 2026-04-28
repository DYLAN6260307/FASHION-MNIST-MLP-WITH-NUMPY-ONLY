from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


FASHION_MNIST_URLS: Dict[str, str] = {
    "train_images": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz",
    "train_labels": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz",
    "test_images": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz",
}

FASHION_MNIST_FILES: Dict[str, str] = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def ensure_fashion_mnist(data_dir: str | Path) -> Path:
    """Download the four Fashion-MNIST IDX gzip files if they are missing."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    for key, filename in FASHION_MNIST_FILES.items():
        target = data_path / filename
        if target.exists() and target.stat().st_size > 0:
            continue
        url = FASHION_MNIST_URLS[key]
        print(f"Downloading {filename} ...")
        urllib.request.urlretrieve(url, target)
    return data_path


def _read_idx_images(path: str | Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image IDX magic number {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n_images, rows, cols)


def _read_idx_labels(path: str | Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label IDX magic number {magic} in {path}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    if labels.shape[0] != n_labels:
        raise ValueError(f"Label count mismatch in {path}: {labels.shape[0]} != {n_labels}")
    return labels.astype(np.int64)


def _standardize_images(images: np.ndarray, flatten: bool = True) -> np.ndarray:
    x = images.astype(np.float32) / 255.0
    mean = np.float32(0.2860)
    std = np.float32(0.3530)
    x = (x - mean) / std
    if flatten:
        x = x.reshape(x.shape[0], -1)
    return x


def load_fashion_mnist(
    data_dir: str | Path = "data/fashion-mnist",
    validation_size: int = 10_000,
    seed: int = 42,
    flatten: bool = True,
    max_train: int | None = None,
    max_val: int | None = None,
    max_test: int | None = None,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load train/validation/test splits.

    The original 60k training images are shuffled deterministically and split
    into train and validation sets. Optional max_* arguments are useful for
    quick smoke tests and hyperparameter search.
    """
    data_path = ensure_fashion_mnist(data_dir)
    train_images = _read_idx_images(data_path / FASHION_MNIST_FILES["train_images"])
    train_labels = _read_idx_labels(data_path / FASHION_MNIST_FILES["train_labels"])
    test_images = _read_idx_images(data_path / FASHION_MNIST_FILES["test_images"])
    test_labels = _read_idx_labels(data_path / FASHION_MNIST_FILES["test_labels"])

    rng = np.random.default_rng(seed)
    indices = rng.permutation(train_images.shape[0])
    val_indices = indices[:validation_size]
    train_indices = indices[validation_size:]

    x_train = _standardize_images(train_images[train_indices], flatten=flatten)
    y_train = train_labels[train_indices]
    x_val = _standardize_images(train_images[val_indices], flatten=flatten)
    y_val = train_labels[val_indices]
    x_test = _standardize_images(test_images, flatten=flatten)
    y_test = test_labels

    if max_train is not None:
        x_train, y_train = x_train[:max_train], y_train[:max_train]
    if max_val is not None:
        x_val, y_val = x_val[:max_val], y_val[:max_val]
    if max_test is not None:
        x_test, y_test = x_test[:max_test], y_test[:max_test]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def unstandardize_flat_images(x: np.ndarray) -> np.ndarray:
    """Convert normalized flattened images back to uint8 28x28 images."""
    images = x.reshape(-1, 28, 28) * 0.3530 + 0.2860
    return np.clip(images * 255.0, 0, 255).astype(np.uint8)

