from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import NormalizationStats


@dataclass(frozen=True)
class Sample:
    patch: torch.Tensor  # (C, H, W) float32
    meta: torch.Tensor  # (M,) float32
    y: torch.Tensor  # () int64
    month: torch.Tensor  # () int64


class PatchMetaDataset(Dataset):
    def __init__(
        self,
        patches: np.ndarray,
        meta: np.ndarray,
        months: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        norm: Optional[NormalizationStats] = None,
        keep_months: Optional[set[int]] = None,
        add_patch_topo_to_meta: bool = True,
        topo_channel_indices: tuple[int, int, int, int] = (10, 11, 12, 13),
        augment: bool = False,
        aug_flip_prob: float = 0.5,
        aug_rot90_prob: float = 0.5,
        aug_noise_std: float = 0.0,
    ) -> None:
        self.patches = patches
        self.meta = meta
        self.months = months
        self.y = y

        if keep_months is not None:
            m = months[indices]
            mask = np.isin(m, np.array(sorted(keep_months), dtype=m.dtype))
            indices = indices[mask]

        self.indices = indices.astype(np.int64)
        self.norm = norm
        self.add_patch_topo_to_meta = add_patch_topo_to_meta
        self.topo_channel_indices = topo_channel_indices
        self.augment = augment
        self.aug_flip_prob = float(aug_flip_prob)
        self.aug_rot90_prob = float(aug_rot90_prob)
        self.aug_noise_std = float(aug_noise_std)

        if norm is not None:
            mean = torch.from_numpy(norm.mean).view(-1, 1, 1)
            std = torch.from_numpy(norm.std).view(-1, 1, 1)
            self._mean = mean
            self._std = std
        else:
            self._mean = None
            self._std = None

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int) -> Sample:
        idx = int(self.indices[i])

        patch_np = self.patches[idx].astype(np.float32)  # (C,H,W)

        if self.augment:
            # H/W flips
            if np.random.rand() < self.aug_flip_prob:
                patch_np = np.flip(patch_np, axis=2)  # flip W
            if np.random.rand() < self.aug_flip_prob:
                patch_np = np.flip(patch_np, axis=1)  # flip H

            # Random rotation by 90 degrees
            if np.random.rand() < self.aug_rot90_prob:
                k = int(np.random.randint(0, 4))
                patch_np = np.rot90(patch_np, k=k, axes=(1, 2))

            # Small Gaussian noise
            if self.aug_noise_std > 0:
                patch_np = patch_np + \
                    np.random.normal(0.0, self.aug_noise_std,
                                     size=patch_np.shape).astype(np.float32)

            # Ensure contiguous after numpy view ops (flip/rot)
            patch_np = np.ascontiguousarray(patch_np)

        patch = torch.from_numpy(patch_np)
        if self._mean is not None and self._std is not None:
            patch = (patch - self._mean) / self._std

        meta = torch.from_numpy(self.meta[idx].astype(np.float32))
        if self.add_patch_topo_to_meta:
            # Topo channels are expected to be in the patch as configured in config.json.
            # Use center pixel to avoid edge effects.
            h = int(patch_np.shape[1])
            w = int(patch_np.shape[2])
            cy = h // 2
            cx = w // 2
            topo = patch_np[list(self.topo_channel_indices),
                            cy, cx].astype(np.float32)
            meta = torch.cat([meta, torch.from_numpy(topo)], dim=0)
        y = torch.tensor(int(self.y[idx]), dtype=torch.int64)
        month = torch.tensor(int(self.months[idx]), dtype=torch.int64)

        return Sample(patch=patch, meta=meta, y=y, month=month)
