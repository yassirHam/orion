"""Evaluation (O1.4): classification metrics + confusion matrix + anomaly AUC."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    accuracy_score,
)

from .data import CLASS_NAMES


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    ys, preds, scores, paths_idx = [], [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        prob = torch.softmax(logits, dim=1)[:, 1]  # P(defective)
        pred = logits.argmax(dim=1).cpu().numpy()
        ys.append(yb.numpy())
        preds.append(pred)
        scores.append(prob.cpu().numpy())
    y = np.concatenate(ys)
    pred = np.concatenate(preds)
    score = np.concatenate(scores)
    return y, pred, score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # Anomaly-detection metric: ROC-AUC on the P(defective) score (needs both classes).
    try:
        auroc = float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else None
    except ValueError:
        auroc = None
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(2)
        },
        "macro": {
            "precision": float(macro_p),
            "recall": float(macro_r),
            "f1": float(macro_f1),
        },
        "confusion_matrix": cm.tolist(),  # rows=true [good,defective], cols=pred
        "auroc": auroc,
    }


def save_confusion_png(cm, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
