"""Split-MNIST: the classic class-incremental protocol.

Five tasks of two digits each (0/1, 2/3, 4/5, 6/7, 8/9), one shared 10-way
head, trained strictly sequentially with no task labels at test time.
MNIST is fetched as raw IDX files (no framework dependency) and cached in
``data/mnist/`` (gitignored).
"""
from __future__ import annotations

import gzip
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _fetch(data_dir: Path) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, name in FILES.items():
        path = data_dir / name
        if not path.exists():
            urllib.request.urlretrieve(MIRROR + name, path)
        paths[key] = path
    return paths


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, = struct.unpack(">I", f.read(4))
        ndim = magic & 0xFF
        shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


@dataclass
class Task:
    name: str
    classes: tuple[int, int]
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def load_split_mnist(data_dir: str = "data/mnist",
                     max_per_class: int | None = None) -> list[Task]:
    paths = _fetch(Path(data_dir))
    xtr = _read_idx(paths["train_images"]).reshape(-1, 784) / 255.0
    ytr = _read_idx(paths["train_labels"])
    xte = _read_idx(paths["test_images"]).reshape(-1, 784) / 255.0
    yte = _read_idx(paths["test_labels"])

    tasks = []
    for a, b in [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]:
        tr = np.isin(ytr, (a, b))
        te = np.isin(yte, (a, b))
        x_train, y_train = xtr[tr].astype(np.float64), ytr[tr].astype(int)
        x_test, y_test = xte[te].astype(np.float64), yte[te].astype(int)
        if max_per_class is not None:
            keep = np.concatenate([
                np.flatnonzero(y_train == c)[:max_per_class] for c in (a, b)])
            x_train, y_train = x_train[keep], y_train[keep]
        tasks.append(Task(f"{a}vs{b}", (a, b), x_train, y_train,
                          x_test, y_test))
    return tasks
