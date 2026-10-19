"""Generate a tiny, deterministic MVTec-AD-style dataset for offline runs.

This lets the baseline run end-to-end with no license-gated download. The
images are synthetic textures: "good" = smooth gradient + mild noise;
"defective" = same, plus an injected bright/dark blob ("scratch"/"spot"),
mirroring MVTec's good-vs-defect structure and folder layout exactly.

For the *real* experiment, point --data-root at an extracted MVTec AD category
instead; the pipeline code is identical.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image


def _good_image(rng: np.random.RandomState, size: int) -> np.ndarray:
    # Smooth diagonal gradient + light gaussian noise.
    x = np.linspace(0, 1, size)
    base = np.add.outer(x, x) / 2.0
    img = base + rng.normal(0, 0.03, size=(size, size))
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def _defect_image(rng: np.random.RandomState, size: int, kind: str) -> np.ndarray:
    img = _good_image(rng, size).astype(np.float32)
    if kind == "spot":  # bright circular blob
        cy, cx = rng.randint(size // 4, 3 * size // 4, size=2)
        r = rng.randint(size // 12, size // 6)
        yy, xx = np.ogrid[:size, :size]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        img[mask] = np.clip(img[mask] + 120, 0, 255)
    else:  # "scratch": dark line
        y = rng.randint(size // 4, 3 * size // 4)
        thickness = rng.randint(1, 3)
        img[y : y + thickness, :] = np.clip(img[y : y + thickness, :] - 120, 0, 255)
    return img.astype(np.uint8)


def _save(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").convert("RGB").save(path)


def generate(root: str, category: str, size: int, seed: int, counts: dict) -> None:
    rng = np.random.RandomState(seed)
    base = Path(root) / category

    for i in range(counts["train_good"]):
        _save(_good_image(rng, size), base / "train" / "good" / f"{i:03d}.png")
    for i in range(counts["test_good"]):
        _save(_good_image(rng, size), base / "test" / "good" / f"{i:03d}.png")
    for i in range(counts["test_spot"]):
        _save(_defect_image(rng, size, "spot"), base / "test" / "spot" / f"{i:03d}.png")
    for i in range(counts["test_scratch"]):
        _save(
            _defect_image(rng, size, "scratch"),
            base / "test" / "scratch" / f"{i:03d}.png",
        )
    print(f"[synth] wrote dataset to {base}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--category", default="synthetic_widget")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-good", type=int, default=120)
    ap.add_argument("--test-good", type=int, default=40)
    ap.add_argument("--test-spot", type=int, default=20)
    ap.add_argument("--test-scratch", type=int, default=20)
    args = ap.parse_args()
    generate(
        args.data_root,
        args.category,
        args.size,
        args.seed,
        {
            "train_good": args.train_good,
            "test_good": args.test_good,
            "test_spot": args.test_spot,
            "test_scratch": args.test_scratch,
        },
    )


if __name__ == "__main__":
    main()
