from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def _ensure_hw(value: object, height: int, width: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == object:
        array = np.stack([np.asarray(item) for item in array], axis=0)
    array = np.asarray(array, dtype=np.float32)

    if array.ndim == 0:
        return np.full((height, width), float(array), dtype=np.float32)
    if array.ndim == 1:
        if array.size == height * width:
            return array.reshape(height, width)
        if array.size == height and height == width:
            return np.tile(array.reshape(height, 1), (1, width))
        raise ValueError(f"Unexpected 1D band shape: {array.shape}")
    if array.ndim == 2:
        if array.shape == (height, width):
            return array
        if array.shape[0] >= height and array.shape[1] >= width:
            start_y = (array.shape[0] - height) // 2
            start_x = (array.shape[1] - width) // 2
            return array[start_y:start_y + height, start_x:start_x + width]
        if array.size == height * width:
            return array.reshape(height, width)
        raise ValueError(f"Unexpected 2D band shape: {array.shape}")
    raise ValueError(f"Unexpected band ndim: {array.ndim}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert per-species Parquet patches to the model's NumPy format."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        required=True,
        help="Training dataset directory containing config.json and class_names.json.",
    )
    args = parser.parse_args()

    with (args.reference_dir / "config.json").open(encoding="utf-8") as file:
        config = json.load(file)
    with (args.reference_dir / "class_names.json").open(encoding="utf-8") as file:
        class_names = json.load(file)

    bands = config["bands"]
    channels, height, width = config["patch_shape"]
    if len(bands) != channels:
        raise ValueError("Config band count does not match patch channel count.")

    files = sorted(args.input_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {args.input_dir}")

    class_to_id = {name: index for index, name in enumerate(class_names)}
    all_patches: list[np.ndarray] = []
    all_meta: list[np.ndarray] = []
    all_months: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    required_columns = set(bands) | {"month", "l3_species"}
    for path in files:
        print(f"Processing {path.name}")
        frame = pd.read_parquet(path)
        if frame.empty:
            raise ValueError(f"{path.name} contains no rows.")
        missing_columns = required_columns.difference(frame.columns)
        if missing_columns:
            raise ValueError(
                f"{path.name} is missing required columns: {sorted(missing_columns)}"
            )

        patches = np.stack([
            np.stack(
                [_ensure_hw(row[band], height, width) for band in bands],
                axis=0,
            ).astype(np.float16)
            for _, row in frame.iterrows()
        ])

        months = frame["month"].to_numpy(dtype=np.int64)
        if np.any((months < 1) | (months > 12)):
            raise ValueError(f"{path.name} contains month values outside 1..12.")
        angle = months / 12.0 * (2.0 * np.pi)
        meta = np.stack([np.sin(angle), np.cos(angle)], axis=1).astype(np.float16)

        species = frame["l3_species"].astype(str)
        unknown_species = sorted(set(species) - class_to_id.keys())
        if unknown_species:
            raise ValueError(f"{path.name} contains unknown species: {unknown_species}")
        labels = species.map(class_to_id).to_numpy(dtype=np.int16)

        all_patches.append(patches)
        all_meta.append(meta)
        all_months.append(months.astype(np.uint8))
        all_labels.append(labels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "patches_f16.npy", np.concatenate(all_patches))
    np.save(args.output_dir / "meta_f16.npy", np.concatenate(all_meta))
    np.save(args.output_dir / "months_u1.npy", np.concatenate(all_months))
    np.save(args.output_dir / "id_y_i2.npy", np.concatenate(all_labels))
    shutil.copyfile(args.reference_dir / "config.json", args.output_dir / "config.json")
    shutil.copyfile(
        args.reference_dir / "class_names.json",
        args.output_dir / "class_names.json",
    )
    print(f"Saved arrays to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
