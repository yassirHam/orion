# ORION O1.6 — Baseline Result & Failure Analysis

All numbers below are from a real run (`seed=42`, synthetic MVTec-layout
stand-in, small CNN, CPU). They are reproducible: the confusion matrix is
identical across repeated runs because every RNG is seeded and the splits are
deterministic.

## Baseline result (established before any fusion)

| Metric            | Value |
|-------------------|-------|
| Test accuracy     | 0.983 |
| Macro F1          | 0.973 |
| AUROC             | 1.000 |
| Operating threshold (val-selected) | 0.405 |

Confusion matrix (threshold-based, rows = true `[good, defective]`, cols = pred):

```
                pred good   pred defective
true good            48            0          (0 false positives)
true defective        1           11          (1 false negative)
```

**This is the Sprint-1 baseline to beat.** AUROC = 1.0 means the anomaly score
*perfectly ranks* defective above good; the only error is one defective image
scoring just under the chosen threshold.

## False negatives (missed defects) — the costly error

* **Count: 1.** Image `test/spot/001.png`, defect type **spot**, anomaly score
  **0.4034**, threshold **0.405** → missed by a margin of 0.0016.
* **Interpretation:** this is a *threshold* problem, not a *separability*
  problem (AUROC is 1.0). A small, hard "spot" defect produced a borderline
  score. In an industrial setting a false negative (shipping a defective part)
  is the expensive error, so the threshold should be biased toward recall.

## False positives (good flagged as defective)

* **Count: 0.** No defect-free image was flagged. Precision on the defective
  class is therefore 1.0 in this run.

## The more important finding: naive supervised ≠ anomaly detection

Running the dataset-native framing (`mode: mvtec_oneclass`, train on *good*
only) produces a very different, instructive result:

```
test accuracy 0.50,  macro F1 0.33,  confusion [[40,0],[40,0]]
```

The classifier predicts **everything "good"** because it never sees a defective
example in training. Even in the `supervised_split` framing, *argmax* alone
collapses to all-good (`argmax confusion [[48,0],[12,0]]`) due to class
imbalance (48 good / 12 defective) — only **thresholding the anomaly score**
(the correct anomaly-detection decision rule) recovers the strong result above.

**Takeaways that justify the next sprint:**
1. For industrial anomaly detection, evaluate with **AUROC + a tuned threshold**,
   not argmax accuracy.
2. A supervised classifier is fundamentally limited by needing labelled defects;
   the dataset is designed for **one-class / unsupervised** methods.
3. Class imbalance must be handled (we added class-weighted loss; thresholding
   did the heavy lifting).

## Limitations

* Results shown use the synthetic stand-in (MVTec AD is license-gated); numbers
  on real MVTec categories will be lower and are the true target. The pipeline
  is identical — only `dataset.root/category` change.
* Binary good-vs-defective only; defect-type classification and pixel-level
  localization are out of Sprint-1 scope.
* Small CNN on CPU; a fine-tuned ResNet18 (`model.name: resnet18`,
  `pretrained: true`) is wired and expected to do better with network access.

## Next steps (Sprint 2)

* Run on $\geq$3 real MVTec categories; report mean ± std over seeds.
* Add a true one-class baseline (e.g. feature-embedding + kNN / PaDiM-style)
  and compare AUROC against this supervised baseline.
* Threshold policy tuned for recall; add pixel-level localization.
* Then — and only then — begin multimodal fusion.
