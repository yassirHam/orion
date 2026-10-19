"""Data pipeline (O1.2) for an MVTec-AD-style industrial anomaly dataset.

Task framing
------------
MVTec AD frames anomaly detection as: train only on *good* (defect-free)
images; at test time, classify each image as good (normal) vs defective
(anomaly). Sprint 1 establishes a supervised **binary classification baseline**
(good vs defective) over the test set's labels, which is the simplest
defensible baseline before moving to true one-class / fusion methods.

Directory layout (identical to MVTec AD, per-category)
------------------------------------------------------
    <root>/<category>/
        train/good/*.png
        test/good/*.png
        test/<defect_type>/*.png        # one or more defect folders

Leakage prevention
------------------
* Splits are made **per source split**: MVTec's own train/test boundary is
  respected (test images never leak into training).
* We derive train/val from the pool deterministically with a fixed seed and a
  fixed sort order, so the same files always land in the same split.
* No image file appears in more than one split (asserted at build time).
"""
from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Label convention: 0 = good/normal, 1 = defective/anomaly.
GOOD, DEFECT = 0, 1
CLASS_NAMES = ["good", "defective"]


@dataclass
class Sample:
    path: str
    label: int
    defect_type: str  # "good" or the defect folder name


def _list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


def discover_samples(root: str, category: str) -> tuple[list[Sample], list[Sample]]:
    """Return (train_pool, test_pool) following the MVTec layout."""
    base = Path(root) / category
    train_pool: list[Sample] = []
    for p in _list_images(base / "train" / "good"):
        train_pool.append(Sample(str(p), GOOD, "good"))

    test_pool: list[Sample] = []
    test_dir = base / "test"
    if test_dir.is_dir():
        for sub in sorted(test_dir.iterdir()):
            if not sub.is_dir():
                continue
            label = GOOD if sub.name == "good" else DEFECT
            for p in _list_images(sub):
                test_pool.append(Sample(str(p), label, sub.name))
    return train_pool, test_pool


def deterministic_split(
    samples: list[Sample], val_fraction: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    """Split a pool into (train, val) deterministically, stratified by label."""
    rng = random.Random(seed)
    by_label: dict[int, list[Sample]] = {}
    for s in sorted(samples, key=lambda x: x.path):  # fixed order
        by_label.setdefault(s.label, []).append(s)

    train, val = [], []
    for label, items in sorted(by_label.items()):
        items = items[:]  # copy
        rng.shuffle(items)
        n_val = int(round(len(items) * val_fraction))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val


def assert_no_leakage(*splits: list[Sample]) -> None:
    seen: set[str] = set()
    for split in splits:
        for s in split:
            assert s.path not in seen, f"leakage: {s.path} in two splits"
            seen.add(s.path)


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    # ImageNet normalization (matches pretrained ResNet).
    norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                norm,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            norm,
        ]
    )


class AnomalyDataset(Dataset):
    def __init__(self, samples: list[Sample], image_size: int, train: bool):
        self.samples = samples
        self.tf = build_transforms(image_size, train)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        return self.tf(img), s.label


def dataset_version(root: str, category: str) -> str:
    """A content hash over the file list + sizes = a dataset 'version' (O1.5)."""
    base = Path(root) / category
    h = hashlib.sha256()
    for p in sorted(base.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(base)).encode())
            h.update(str(p.stat().st_size).encode())
    return h.hexdigest()[:12]


def class_distribution(samples: list[Sample]) -> dict:
    dist: dict[str, int] = {}
    for s in samples:
        dist[CLASS_NAMES[s.label]] = dist.get(CLASS_NAMES[s.label], 0) + 1
    return dist


def defect_distribution(samples: list[Sample]) -> dict:
    dist: dict[str, int] = {}
    for s in samples:
        dist[s.defect_type] = dist.get(s.defect_type, 0) + 1
    return dict(sorted(dist.items()))
