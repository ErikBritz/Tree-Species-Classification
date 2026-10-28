from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class Metrics:
    macro_f1: float
    per_class_f1: Dict[str, float]
    confusion: np.ndarray


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Tuple[str, ...],
) -> Metrics:
    from sklearn.metrics import f1_score, confusion_matrix

    labels = np.arange(len(class_names))
    macro_f1 = float(
        f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    f1s = f1_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class_f1 = {class_names[i]: float(
        f1s[i]) for i in range(len(class_names))}
    confusion = confusion_matrix(
        y_true, y_pred, labels=np.arange(len(class_names)))
    return Metrics(macro_f1=macro_f1, per_class_f1=per_class_f1, confusion=confusion)


def classification_report_text(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Tuple[str, ...],
) -> str:
    from sklearn.metrics import classification_report

    return classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=list(class_names),
        digits=4,
        zero_division=0,
    )


def save_confusion_plot(
    confusion: np.ndarray,
    class_names: Tuple[str, ...],
    out_path: str,
    normalize: bool = True,
    title: str = "Confusion matrix",
) -> None:
    import matplotlib.pyplot as plt

    cm = confusion.astype(np.float32)
    if normalize:
        denom = cm.sum(axis=1, keepdims=True)
        denom = np.where(denom == 0, 1.0, denom)
        cm = cm / denom

    fig = plt.figure(figsize=(10, 8), dpi=140)
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    ax.set_ylabel("True")
    ax.set_xlabel("Pred")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
