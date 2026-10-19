# ORION — Sprint 1 (AI Research)

A **reproducible industrial-anomaly vision baseline** in PyTorch. Sprint 1
establishes a defensible dataset choice (MVTec AD) and a baseline that can be
**rerun from recorded configuration**, with documented evaluation and failure
cases — before any multimodal fusion.

> **Sprint 1 completion criterion:** *the baseline can be rerun from recorded
> configuration, and its evaluation results and failure cases are documented.*

## Layout

```
orion/
├── orion/
│   ├── data.py            # MVTec-layout dataset, deterministic leakage-free splits (O1.2)
│   ├── synth_data.py      # offline, deterministic MVTec-layout generator
│   ├── model.py           # ResNet18 fine-tune / small CNN (O1.3)
│   ├── evaluate.py        # precision/recall/F1, confusion matrix, AUROC (O1.4)
│   └── reproducibility.py # seeds, git commit, hardware, timing, config hash (O1.5)
├── scripts/train.py       # end-to-end runner → runs/<ts>_<hash>/
├── configs/baseline.yaml  # the experiment config
├── tests/                 # pipeline tests (splits, no leakage)
└── docs/
    ├── dataset.md         # O1.1 dataset selection
    └── failure_analysis.md# O1.6 baseline result + FP/FN analysis
```

## Setup

```bash
pip install -r requirements.txt      # torch/torchvision CPU is fine
```

## Run the baseline (reproducible, offline)

```bash
# 1) generate the deterministic MVTec-layout stand-in
python -m orion.synth_data --data-root data --category synthetic_widget

# 2) train + evaluate; writes a self-contained run folder
python -m scripts.train --config configs/baseline.yaml
```

To run on **real MVTec AD**: download/extract a category, point
`dataset.root`/`dataset.category` at it (same code), and set
`model.name: resnet18`, `model.pretrained: true` if you have network access.

## What a run records (O1.5)

Each run creates `runs/<timestamp>_<confighash>/` with:

* `config.yaml` — resolved configuration
* `manifest.json` — seed, **dataset version hash**, **git commit**, hardware,
  training time, per-split class distributions
* `model.pt` — best checkpoint (by validation accuracy)
* `metrics.json` — accuracy, per-class P/R/F1, macro, confusion matrix, AUROC,
  threshold
* `confusion_matrix.png`
* `predictions.csv` — per-sample `y_true,y_pred,score,path` (drives failure analysis)

## Baseline result (seed=42, synthetic stand-in, small CNN, CPU)

| Metric | Value |
|--------|-------|
| Accuracy | 0.983 |
| Macro F1 | 0.973 |
| AUROC | 1.000 |
| Confusion (thresholded) | `[[48,0],[1,11]]` |
| Training time | ~1.4 s (CPU) |

See [`docs/failure_analysis.md`](docs/failure_analysis.md) for the FP/FN
inspection and the key finding (naive supervised classification ≠ anomaly
detection; evaluate with AUROC + a tuned threshold).

## Tests

```bash
pytest -q        # verifies deterministic, stratified, leakage-free splits
```

## Reproducibility checklist (O1.5)

- [x] Fixed seeds for Python/NumPy/PyTorch (+ deterministic cuDNN)
- [x] Deterministic, stratified, **leakage-free** splits (asserted)
- [x] Dataset version hash recorded per run
- [x] Git commit recorded per run
- [x] Config saved per run; config hash in the run name
- [x] Model checkpoint saved
- [x] Training time + hardware recorded
