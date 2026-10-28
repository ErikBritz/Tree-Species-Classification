from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import json
import numpy as np


@dataclass(frozen=True)
class DatasetConfig:
    bands: Tuple[str, ...]
    patch_shape: Tuple[int, int, int]  # (C, H, W)
    num_rows: int
    num_ids: int
    num_classes: int


def load_config(dataset_dir: str | Path) -> DatasetConfig:
    dataset_dir = Path(dataset_dir)
    with (dataset_dir / "config.json").open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return DatasetConfig(
        bands=tuple(raw["bands"]),
        patch_shape=tuple(raw["patch_shape"]),
        num_rows=int(raw["num_rows"]),
        num_ids=int(raw["num_ids"]),
        num_classes=int(raw["num_classes"]),
    )


def load_class_names(dataset_dir: str | Path) -> Tuple[str, ...]:
    dataset_dir = Path(dataset_dir)
    with (dataset_dir / "class_names.json").open("r", encoding="utf-8") as f:
        names = json.load(f)
    return tuple(names)


@dataclass(frozen=True)
class Arrays:
    patches: np.ndarray  # (N, C, H, W) float16
    meta: np.ndarray  # (N, 2) float16: [sin_month, cos_month]
    months: np.ndarray  # (N,) uint8 1..12
    y: np.ndarray  # (N,) int16


def load_arrays(dataset_dir: str | Path, mmap: bool = True) -> Arrays:
    dataset_dir = Path(dataset_dir)
    mmap_mode = "r" if mmap else None
    patches = np.load(dataset_dir / "patches_f16.npy", mmap_mode=mmap_mode)
    meta = np.load(dataset_dir / "meta_f16.npy", mmap_mode=mmap_mode)
    months = np.load(dataset_dir / "months_u1.npy", mmap_mode=mmap_mode)
    y = np.load(dataset_dir / "id_y_i2.npy", mmap_mode=mmap_mode)

    return Arrays(patches=patches, meta=meta, months=months, y=y)


def compute_class_counts(y: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y.astype(np.int64), minlength=num_classes)
    return counts.astype(np.int64)


def make_stratified_split(
    y: np.ndarray,
    val_size: float,
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns train_idx, val_idx, test_idx."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=y
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=val_size / (1.0 - test_size),
        random_state=seed,
        stratify=y[train_idx],
    )
    return train_idx, val_idx, test_idx


@dataclass(frozen=True)
class NormalizationStats:
    mean: np.ndarray  # (C,)
    std: np.ndarray  # (C,)


def compute_patch_norm_stats(
    patches: np.ndarray,
    train_idx: np.ndarray,
    eps: float = 1e-6,
    max_samples: Optional[int] = 200_000,
    seed: int = 0,
) -> NormalizationStats:
    """Computes per-band mean/std over train set, streaming to keep memory bounded.

    If max_samples is set, uniformly subsamples train_idx to speed up.
    """
    rng = np.random.default_rng(seed)
    idx = train_idx
    if max_samples is not None and len(idx) > max_samples:
        idx = rng.choice(idx, size=max_samples, replace=False)

    c = patches.shape[1]
    total = np.zeros((c,), dtype=np.float64)
    total_sq = np.zeros((c,), dtype=np.float64)
    n = 0

    # stream in chunks to avoid large temporary arrays
    chunk = 4096
    for start in range(0, len(idx), chunk):
        sel = idx[start: start + chunk]
        x = patches[sel].astype(np.float32)  # (B, C, H, W)
        x = x.reshape(x.shape[0], c, -1)
        total += np.sum(x, axis=(0, 2), dtype=np.float64)
        total_sq += np.sum(x * x, axis=(0, 2), dtype=np.float64)
        n += x.shape[0] * x.shape[2]

    mean = (total / n).astype(np.float32)
    var = (total_sq / n - mean.astype(np.float64) ** 2).astype(np.float32)
    std = np.sqrt(np.maximum(var, 0.0)).astype(np.float32)
    std = np.maximum(std, eps)
    return NormalizationStats(mean=mean, std=std)


def save_norm_stats(path: str | Path, stats: NormalizationStats) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=stats.mean, std=stats.std)


def load_norm_stats(path: str | Path) -> NormalizationStats:
    d = np.load(path)
    return NormalizationStats(mean=d["mean"].astype(np.float32), std=d["std"].astype(np.float32))
