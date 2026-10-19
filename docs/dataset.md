# ORION O1.1 — Dataset Selection

## Chosen dataset: MVTec AD (Anomaly Detection)

**Source.** MVTec Software GmbH — *The MVTec Anomaly Detection Dataset
(MVTec AD)*, Bergmann et al., CVPR 2019.
Project page: <https://www.mvtec.com/company/research/datasets/mvtec-ad>.

**Why this dataset.** It is the de-facto standard benchmark for *industrial*
visual anomaly detection: 15 object/texture categories photographed in a
factory-inspection style, with pixel-accurate ground truth. It directly matches
ORION's domain (industrial defect detection) and is widely cited, so results
are comparable to published baselines.

**License.** CC BY-NC-SA 4.0 (Attribution–NonCommercial–ShareAlike). It is free
for **research/academic** use (our case) with attribution; commercial use is not
permitted. We only use it for research and cite the authors.

**Labels.** Per category, images are organised as:
* `train/good/` — defect-free images only (the dataset is designed for
  one-class / unsupervised anomaly detection: you train on "good" only).
* `test/good/` — defect-free test images.
* `test/<defect_type>/` — defective test images grouped by defect type
  (e.g. *scratch*, *bent*, *contamination*, …), with pixel masks in
  `ground_truth/`.

For Sprint 1 we reduce labels to a **binary** problem: `good` (0) vs
`defective` (1).

**Class distribution.** Varies per category; test sets are intentionally
imbalanced toward defective images across all defect types, while training is
100% good. (Exact per-category counts are recorded automatically in each run's
`manifest.json` under `test_class_distribution` / `test_defect_distribution`.)

**Intended evaluation split.** MVTec ships a fixed `train` / `test` boundary;
we respect it (no test image is ever used for training). We derive a
validation set deterministically from the training pool. We report two framings:

1. **`mvtec_oneclass` (dataset-native):** train on `train/good` only, evaluate
   on the full `test` set. This is the correct long-term framing and exposes
   why a naive supervised classifier fails (see failure analysis).
2. **`supervised_split` (Sprint-1 baseline):** pool all labelled images and make
   stratified, leakage-free train/val/test splits so both classes appear in
   training — a fair supervised-classification baseline before one-class/fusion
   methods.

Primary metric for the anomaly task: **AUROC** on the anomaly score, plus
precision/recall/F1 and a confusion matrix at a validation-selected threshold.

## Reproducible stand-in used in this repo

MVTec AD is license-gated and multi-GB, so it is **not** committed here. To keep
the baseline runnable offline and byte-for-byte reproducible, this repo includes
a deterministic **synthetic generator** (`orion/synth_data.py`) that writes the
*exact MVTec folder layout* (`train/good`, `test/good`, `test/<defect>`) with
`good` vs injected-defect (`spot`, `scratch`) images.

**The pipeline code is identical for both.** To run on the real dataset:

```bash
# download + extract MVTec AD, then:
python -m scripts.train --config configs/baseline.yaml \
    # with dataset.root / dataset.category pointed at e.g. mvtec/bottle
```

All dataset-dependent facts (version hash, counts, distributions) are recorded
per-run in `manifest.json`, so the exact data behind any result is always
traceable.
