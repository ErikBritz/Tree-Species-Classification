from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .data import (
    NormalizationStats,
    compute_class_counts,
    compute_patch_norm_stats,
    load_arrays,
    load_class_names,
    load_config,
    make_stratified_split,
    save_norm_stats,
)
from .metrics import classification_report_text, compute_metrics, save_confusion_plot
from .model import CNNMetaClassifier
from .torch_dataset import PatchMetaDataset


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: list[int] = []
    ps: list[int] = []
    for batch in loader:
        patch = batch.patch.to(device)
        meta = batch.meta.to(device)
        y = batch.y.cpu().numpy()
        logits = model(patch, meta)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        ys.append(y)
        ps.append(pred)
    return np.concatenate(ys), np.concatenate(ps)


def parse_months(s: Optional[str]) -> Optional[set[int]]:
    if s is None or s.strip() == "":
        return None
    months: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        month = int(part)
        if month < 1 or month > 12:
            raise ValueError(f"Month must be in the range 1..12, got {month}.")
        months.add(month)
    return months


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Train CNN+metadata model on TreeSatAI patches")
    p.add_argument("--dataset-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="runs/run1")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=5,
                   help="Early stopping patience on val macro F1")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--use-sampler", action="store_true",
                   help="Use WeightedRandomSampler for imbalance")
    p.add_argument("--no-class-weights", action="store_true",
                   help="Disable class-weighted loss")
    p.add_argument(
        "--class-weight-power",
        type=float,
        default=1.0,
        help="Exponent for inverse-frequency class weights (1.0=inverse freq, 0.5=sqrt, 0=uniform)",
    )
    p.add_argument("--augment", action="store_true",
                   help="Enable random flips, rotations, and optional noise for training patches")
    p.add_argument("--aug-flip-prob", type=float, default=0.5)
    p.add_argument("--aug-rot90-prob", type=float, default=0.5)
    p.add_argument("--aug-noise-std", type=float, default=0.0)
    p.add_argument(
        "--test-only",
        action="store_true",
        help="Skip training and only run test/inference using best.pt checkpoint."
    )

    p.add_argument(
        "--test-months",
        type=str,
        default=None,
        help="Comma list of months to keep in TEST (e.g. '3,4,5,6,7,8,9,10,11')",
    )
    args = p.parse_args(argv)
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        p.error("--epochs, --patience, and --batch-size must be positive.")
    if not 0.0 <= args.aug_flip_prob <= 1.0:
        p.error("--aug-flip-prob must be between 0 and 1.")
    if not 0.0 <= args.aug_rot90_prob <= 1.0:
        p.error("--aug-rot90-prob must be between 0 and 1.")
    if args.aug_noise_std < 0:
        p.error("--aug-noise-std must not be negative.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    best_path = out_dir / "best.pt"
    cfg = load_config(args.dataset_dir)
    class_names = load_class_names(args.dataset_dir)
    arrays = load_arrays(args.dataset_dir, mmap=True)

    train_idx, val_idx, test_idx = make_stratified_split(
        arrays.y.astype(np.int64),
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    checkpoint = None
    if args.test_only:
        if not best_path.is_file():
            raise FileNotFoundError(
                f"Cannot run --test-only without a checkpoint at {best_path}")
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        if checkpoint["class_names"] != list(class_names):
            raise ValueError("Checkpoint class names do not match the dataset.")
        norm = NormalizationStats(
            mean=np.asarray(checkpoint["norm_mean"], dtype=np.float32),
            std=np.asarray(checkpoint["norm_std"], dtype=np.float32),
        )
    else:
        norm = compute_patch_norm_stats(arrays.patches, train_idx, seed=args.seed)
        save_norm_stats(out_dir / "norm_stats.npz", norm)

    keep_test_months = parse_months(args.test_months)

    train_ds = PatchMetaDataset(
        patches=arrays.patches,
        meta=arrays.meta,
        months=arrays.months,
        y=arrays.y,
        indices=train_idx,
        norm=norm,
        augment=args.augment,
        aug_flip_prob=args.aug_flip_prob,
        aug_rot90_prob=args.aug_rot90_prob,
        aug_noise_std=args.aug_noise_std,
    )
    val_ds = PatchMetaDataset(
        patches=arrays.patches,
        meta=arrays.meta,
        months=arrays.months,
        y=arrays.y,
        indices=val_idx,
        norm=norm,
    )
    test_ds = PatchMetaDataset(
        patches=arrays.patches,
        meta=arrays.meta,
        months=arrays.months,
        y=arrays.y,
        indices=test_idx,
        norm=norm,
        keep_months=keep_test_months,
    )
    if len(test_ds) == 0:
        raise ValueError("The test split is empty after filtering by --test-months.")

    # Imbalance handling
    class_counts = compute_class_counts(
        arrays.y[train_idx].astype(np.int64), cfg.num_classes)
    inv = (class_counts.sum() / np.maximum(class_counts, 1)).astype(np.float32)
    power = float(args.class_weight_power)
    if power <= 0:
        class_weights = np.ones_like(inv, dtype=np.float32)
    else:
        class_weights = np.power(inv, power).astype(np.float32)
        class_weights = class_weights / class_weights.mean()  # keep loss scale stable

    sampler = None
    shuffle = True
    if args.use_sampler:
        # per-sample weights based on class frequency
        y_train = arrays.y[train_idx].astype(np.int64)
        w = class_weights[y_train]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(w),
            num_samples=len(w),
            replacement=True,
        )
        shuffle = False

    def collate(batch):
        patch = torch.stack([b.patch for b in batch], dim=0)
        meta = torch.stack([b.meta for b in batch], dim=0)
        y = torch.stack([b.y for b in batch], dim=0)
        month = torch.stack([b.month for b in batch], dim=0)
        return type(batch[0])(patch=patch, meta=meta, y=y, month=month)

    def worker_init_fn(worker_id: int) -> None:
        # Make numpy-based augmentation reproducible across DataLoader workers.
        np.random.seed(args.seed + worker_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate,
        worker_init_fn=worker_init_fn if args.num_workers and args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Meta dim can differ from arrays.meta.shape[1] if the Dataset augments metadata
    # (e.g., by extracting topo channels from the patch).
    meta_dim = int(train_ds[0].meta.numel())

    model = CNNMetaClassifier(
        in_channels=cfg.patch_shape[0],
        num_classes=cfg.num_classes,
        meta_dim=meta_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_weight = None
    if not args.no_class_weights:
        loss_weight = torch.from_numpy(class_weights).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=loss_weight)

    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    best_macro = -1.0
    history = []
    epochs_no_improve = 0

    if not args.test_only:
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            n = 0

            pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
            for batch in pbar:
                patch = batch.patch.to(device, non_blocking=True)
                meta = batch.meta.to(device, non_blocking=True)
                y = batch.y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    logits = model(patch, meta)
                    loss = loss_fn(logits, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                bs = int(y.shape[0])
                total_loss += float(loss.item()) * bs
                n += bs
                pbar.set_postfix(loss=total_loss / max(n, 1))

            yv, pv = predict(model, val_loader, device)
            val_metrics = compute_metrics(yv, pv, class_names)

            row = {
                "epoch": epoch,
                "train_loss": total_loss / max(n, 1),
                "val_macro_f1": val_metrics.macro_f1,
            }
            history.append(row)

            print(f"val macro F1: {val_metrics.macro_f1:.4f}")

            if val_metrics.macro_f1 > best_macro:
                best_macro = val_metrics.macro_f1
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": asdict(cfg),
                        "class_names": list(class_names),
                        "norm_mean": norm.mean,
                        "norm_std": norm.std,
                        "args": vars(args),
                    },
                    best_path,
                )
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= args.patience:
                print(
                    f"Early stopping: no val macro F1 improvement for {args.patience} epochs")
                break

    # Load best checkpoint
    ckpt = checkpoint or torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    yt, pt = predict(model, test_loader, device)

    test_metrics = compute_metrics(yt, pt, class_names)

    # Additional metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    accuracy = float(accuracy_score(yt, pt))
    labels = np.arange(len(class_names))
    precision = precision_score(
        yt, pt, labels=labels, average=None, zero_division=0
    )
    recall = recall_score(yt, pt, labels=labels, average=None, zero_division=0)
    per_class_precision = {class_names[i]: float(
        precision[i]) for i in range(len(class_names))}
    per_class_recall = {class_names[i]: float(
        recall[i]) for i in range(len(class_names))}

    report = classification_report_text(yt, pt, class_names)
    print("\n=== TEST CLASSIFICATION REPORT ===\n")
    print(report)
    print(f"TEST macro F1: {test_metrics.macro_f1:.4f}")

    # Save outputs
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "test_report.txt").write_text(report, encoding="utf-8")

    correct_mask = yt == pt
    per_class_correct = {
        name: int(np.sum(correct_mask & (yt == i)))
        for i, name in enumerate(class_names)
    }

    test_out = {
        "accuracy": accuracy,
        "macro_f1": test_metrics.macro_f1,
        "per_class_f1": test_metrics.per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_correct": per_class_correct,
    }
    (out_dir / "test_metrics.json").write_text(json.dumps(test_out, indent=2), encoding="utf-8")

    np.save(out_dir / "confusion.npy", test_metrics.confusion)
    save_confusion_plot(
        test_metrics.confusion,
        class_names,
        str(out_dir / "confusion.png"),
        normalize=True,
        title="Confusion (row-normalized)",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
