import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a prepared dataset.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    args = parser.parse_args()

    names = (
        "patches_f16.npy",
        "meta_f16.npy",
        "months_u1.npy",
        "id_y_i2.npy",
        "id_ptr_i8.npy",
    )
    for name in names:
        path = args.dataset_dir / name
        if not path.exists():
            print(f"{name}: not present")
            continue
        array = np.load(path, mmap_mode="r")
        print(f"{name}: shape={array.shape}, dtype={array.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
