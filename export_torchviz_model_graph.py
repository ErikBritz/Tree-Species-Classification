from __future__ import annotations

import json
import argparse
from pathlib import Path
import shutil

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export the CNN + metadata model's TorchViz graph."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("training_results"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset config
    with (args.dataset_dir / "config.json").open(encoding="utf-8") as file:
        cfg = json.load(file)

    in_channels = cfg["patch_shape"][0]
    num_classes = int(cfg["num_classes"])

    # Compute meta_dim the same way training does (Dataset may append topo channels)
    from src.torch_dataset import PatchMetaDataset

    patches = np.load(args.dataset_dir / "patches_f16.npy", mmap_mode="r")
    meta = np.load(args.dataset_dir / "meta_f16.npy", mmap_mode="r")
    months = np.load(args.dataset_dir / "months_u1.npy", mmap_mode="r")
    y = np.load(args.dataset_dir / "id_y_i2.npy", mmap_mode="r")

    ds = PatchMetaDataset(
        patches=patches,
        meta=meta,
        months=months,
        y=y,
        indices=np.array([0], dtype=np.int64),
        norm=None,
        add_patch_topo_to_meta=True,
    )
    sample = ds[0]
    meta_dim = int(sample.meta.numel())

    # Build full model
    from src.model import CNNMetaClassifier

    model = CNNMetaClassifier(
        in_channels=int(in_channels),
        num_classes=num_classes,
        meta_dim=meta_dim,
    )
    model.eval()

    # Use real-shaped dummy inputs
    patch = sample.patch.unsqueeze(0)  # (1, C, H, W)
    meta_in = sample.meta.unsqueeze(0)  # (1, M)

    # IMPORTANT: do NOT use torch.no_grad() here.
    # TorchViz builds the graph from autograd metadata (grad_fn).
    logits = model(patch, meta_in)

    # TorchViz graph
    from torchviz import make_dot

    dot = make_dot(
        logits,
        params=dict(model.named_parameters()),
    )

    base = args.out_dir / "cnn_meta_torchviz"

    # Always save the Graphviz source
    gv_path = base.with_suffix(".gv")
    gv_path.write_text(dot.source, encoding="utf-8")

    # Try to render to PNG (requires Graphviz 'dot' installed on the system)
    dot_exe = shutil.which("dot")
    if dot_exe is None:
        print("Graphviz 'dot' not found on PATH; skipping PNG render.")
        print(f"Wrote: {gv_path}")
        print("Install Graphviz and ensure 'dot' is on PATH to render PNG.")
    else:
        dot.format = "png"
        dot.render(str(base), cleanup=True)
        print(f"Wrote: {base.with_suffix('.png')}")

    # Also print key shapes for sanity
    print(f"Input patch: {tuple(patch.shape)}")
    print(f"Input meta:  {tuple(meta_in.shape)}")
    print(f"Output logits: {tuple(logits.shape)}")
    print(f"meta_dim inferred: {meta_dim}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
