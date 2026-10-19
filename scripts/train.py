"""ORION Sprint 1 baseline runner (O1.3 + O1.4 + O1.5).

Trains a good-vs-defective classifier on an MVTec-AD-style dataset, evaluates
with precision/recall/F1 + confusion matrix + AUROC, and writes a fully
reproducible run folder:

    runs/<timestamp>_<confighash>/
        config.yaml            resolved config
        manifest.json          seed, dataset version, git commit, hardware, timing
        model.pt               best checkpoint (by val accuracy)
        metrics.json           test metrics
        confusion_matrix.png   plot
        predictions.csv        per-sample y_true,y_pred,score,path (for failure analysis)

Usage:
    python -m scripts.train --config configs/baseline.yaml
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

# allow running as `python -m scripts.train` or `python scripts/train.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.data import (  # noqa: E402
    AnomalyDataset,
    assert_no_leakage,
    class_distribution,
    dataset_version,
    defect_distribution,
    deterministic_split,
    discover_samples,
)
from orion.evaluate import (  # noqa: E402
    collect_predictions,
    compute_metrics,
    save_confusion_png,
)
from orion.model import build_model  # noqa: E402
from orion.reproducibility import (  # noqa: E402
    Timer,
    config_hash,
    git_commit,
    hardware_info,
    set_seed,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, optim, loss_fn, device):
    model.train()
    total, correct, running = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optim.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optim.step()
        running += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return running / total, correct / total


@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    total, correct = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        correct += (model(xb).argmax(1) == yb).sum().item()
        total += xb.size(0)
    return correct / max(total, 1)


def _select_threshold(y_val, score_val):
    """Pick the score threshold that maximizes F1 on the validation set."""
    import numpy as np
    from sklearn.metrics import f1_score

    if len(set(y_val.tolist())) < 2:
        return 0.5
    candidates = np.unique(np.concatenate([[0.0, 1.0], score_val]))
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        f1 = f1_score(y_val, (score_val >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = cfg["train"]["seed"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = cfg["dataset"]["root"]
    category = cfg["dataset"]["category"]

    # ---- data (O1.2) --------------------------------------------------------
    train_pool, test_pool = discover_samples(root, category)
    if not train_pool or not test_pool:
        print(
            f"ERROR: no data under {root}/{category}. "
            f"Generate the synthetic set first:\n"
            f"  python -m orion.synth_data --data-root {root} --category {category}",
            file=sys.stderr,
        )
        return 2

    mode = cfg["dataset"].get("mode", "mvtec_oneclass")
    if mode == "supervised_split":
        # Fair *supervised classification* baseline: pool all labeled images
        # (train/good + the labeled test set) and make stratified, leakage-free
        # train/val/test splits so both classes appear in training.
        all_labeled = train_pool + test_pool
        trainval, test_s = deterministic_split(all_labeled, 0.3, seed)
        train_s, val_s = deterministic_split(
            trainval, cfg["dataset"]["val_fraction"], seed + 1
        )
    else:
        # MVTec one-class framing: train on good only; evaluate on the test set.
        train_s, val_s = deterministic_split(
            train_pool, cfg["dataset"]["val_fraction"], seed
        )
        test_s = test_pool
    assert_no_leakage(train_s, val_s, test_s)

    img = cfg["dataset"]["image_size"]
    bs = cfg["train"]["batch_size"]
    g = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(AnomalyDataset(train_s, img, True), batch_size=bs,
                          shuffle=True, generator=g)
    val_dl = DataLoader(AnomalyDataset(val_s, img, False), batch_size=bs)
    test_dl = DataLoader(AnomalyDataset(test_s, img, False), batch_size=bs)

    # ---- model (O1.3) -------------------------------------------------------
    model = build_model(
        cfg["model"]["name"], num_classes=2, pretrained=cfg["model"]["pretrained"]
    ).to(device)
    optim = torch.optim.Adam(
        model.parameters(), lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    # Class-weighted loss to counter the good/defective imbalance.
    counts = [0, 0]
    for s in train_s:
        counts[s.label] += 1
    weights = torch.tensor(
        [sum(counts) / (2 * c) if c > 0 else 0.0 for c in counts],
        dtype=torch.float32, device=device,
    )
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    # ---- run folder (O1.5) --------------------------------------------------
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg["output"]["dir"]) / f"{ts}_{config_hash(cfg)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # ---- train --------------------------------------------------------------
    best_val, best_state = -1.0, None
    history = []
    with Timer("train") as t:
        for epoch in range(cfg["train"]["epochs"]):
            tr_loss, tr_acc = train_one_epoch(model, train_dl, optim, loss_fn, device)
            va = eval_acc(model, val_dl, device)
            history.append({"epoch": epoch, "train_loss": tr_loss,
                            "train_acc": tr_acc, "val_acc": va})
            print(f"epoch {epoch}: train_loss={tr_loss:.4f} "
                  f"train_acc={tr_acc:.3f} val_acc={va:.3f}")
            if va > best_val:
                best_val = va
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(best_state, run_dir / "model.pt")

    # ---- evaluate (O1.4) ----------------------------------------------------
    # Choose an operating threshold on the anomaly score using the validation
    # set (standard for anomaly detection); fall back to 0.5 if val is single-class.
    vy, _, vscore = collect_predictions(model, val_dl, device)
    threshold = _select_threshold(vy, vscore)

    y, pred_argmax, score = collect_predictions(model, test_dl, device)
    pred = (score >= threshold).astype(int)  # thresholded anomaly decision
    metrics = compute_metrics(y, pred, score)
    metrics["threshold"] = float(threshold)
    metrics["argmax_confusion_matrix"] = compute_metrics(
        y, pred_argmax, score
    )["confusion_matrix"]
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    save_confusion_png(metrics["confusion_matrix"], str(run_dir / "confusion_matrix.png"))

    # per-sample predictions for failure analysis (O1.6)
    with open(run_dir / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "defect_type", "y_true", "y_pred", "score_defective"])
        for s, yt, yp, sc in zip(test_s, y, pred, score):
            w.writerow([s.path, s.defect_type, int(yt), int(yp), f"{sc:.4f}"])

    # ---- manifest (O1.5) ----------------------------------------------------
    manifest = {
        "timestamp": ts,
        "seed": seed,
        "config_hash": config_hash(cfg),
        "dataset": {
            "root": root,
            "category": category,
            "version": dataset_version(root, category),
            "mode": mode,
            "train_pool": len(train_pool),
            "test_pool": len(test_pool),
            "split_sizes": {"train": len(train_s), "val": len(val_s),
                            "test": len(test_s)},
            "train_class_distribution": class_distribution(train_s),
            "val_class_distribution": class_distribution(val_s),
            "test_class_distribution": class_distribution(test_s),
            "test_defect_distribution": defect_distribution(test_s),
        },
        "git_commit": git_commit(),
        "hardware": hardware_info(),
        "training_seconds": round(t.seconds, 2),
        "best_val_acc": best_val,
        "history": history,
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== RESULT ===")
    print(f"run dir     : {run_dir}")
    print(f"test acc    : {metrics['accuracy']:.3f}")
    print(f"macro F1    : {metrics['macro']['f1']:.3f}")
    print(f"AUROC       : {metrics['auroc']}")
    print(f"confusion   : {metrics['confusion_matrix']}  (rows=true[good,def], cols=pred)")
    print(f"train time  : {t.seconds:.2f}s on {manifest['hardware']['gpu']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
