# Mobile Vision-Transformer Benchmark for Fine-Grained Vietnamese Medicinal Plant Classification

Reproducibility artifact — notebooks, conversion/deployment scripts, and training outputs
(LiteRT/TFLite models, metrics CSVs and JSONs) accompanying our benchmarking study of
compact CNN and vision-transformer backbones for on-device medicinal plant recognition.

<!-- Fill these in once the archive is minted -->
[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey.svg)](https://doi.org/XX.XXXX/zenodo.XXXXXXX)
**Archive DOI:** `10.XXXX/zenodo.XXXXXXX` (Zenodo/Figshare — pending)
**Dataset DOI (Viet Medi Species 2026):** `10.XXXX/zenodo.XXXXXXX` (pending)

---

## Overview

This repository contains everything needed to reproduce and inspect the results of a benchmark
comparing three compact image-classification backbones for **fine-grained classification of 2,721
Vietnamese medicinal plant classes**, and to reproduce the six models we deployed and profiled on
physical Android devices.

The study evaluates each backbone along two axes:

1. **Accuracy** under four image-preprocessing pipelines (`baseline`, `clahe`, `sobel`,
   `clahe_sobel`), reported as 3-seed mean ± std with paired-significance testing.
2. **On-device cost** — model size, cold-start time, inference latency, peak RAM, and battery
   efficiency — measured on three physical Android phones through the Google ML Kit image
   classifier.

The repository is intended as a permanent, citable companion to the manuscript. It is **code and
results only**; the underlying image data is distributed separately as the *Viet Medi Species 2026*
dataset (see [Dataset](#dataset)).

---

## Repository structure

```
.
├── notebooks_n_python_programs/   # training notebooks + conversion/deployment/analysis scripts
├── results_bundle_final/          # training outputs: TFLite models, metrics CSVs & JSONs, labels
├── mobile_benchmarking.csv        # on-device measurements (size, latency, RAM, battery) per device × model
└── REPRO_AND_FACTS.md             # frozen record of environments, splits, params, and known caveats
```

### `notebooks_n_python_programs/`

| File | Purpose |
|------|---------|
| `01-full-vit.ipynb` | Trains the ViT-B/16 reference model that establishes the per-preprocessing accuracy ceiling. |
| `02-mobile-transformers.ipynb` | Trains and evaluates the three mobile backbones across the four preprocessing pipelines and seeds. |
| `run_grid.sh` | Driver that runs the full architecture × preprocessing × seed training grid. |
| `aggregate_preprocessing.py` | Aggregates per-run metrics into `preprocessing_comparison.csv` and computes paired significance (`preprocessing_significance.csv`). |
| `export_tflite.py` | Baseline PyTorch → LiteRT/TFLite export for the CNN backbones. |
| `export_softmax_tflite.py` | Appends a softmax head so ML Kit receives a probability distribution (monotonic; argmax/top-k unchanged). |
| `export_mobilevit_litert.py` | LiteRT export path for MobileViT. |
| `reexport_mobilevit_nhwc.py` | Wraps MobileViT (NCHW) into an NHWC `[1,224,224,3]` graph required by ML Kit. |
| `write_metadata.py` | Embeds per-model `NormalizationOptions` (ImageNet stats for the CNNs; plain `/255` for MobileViT). |
| `make_labels.py` | Emits `labels.txt` in class-index order (sorted class-folder names, filtered to ≥25 images). |
| `convert_all.sh` | Runs the full export → NHWC → metadata pipeline to produce the shipped models. |
| `diagnose_tflite.py` | Inspects a `.tflite` graph (I/O tensors, dtypes, ops) for troubleshooting. |
| `parity_check_deployed.py` | Verifies deployed fp16/fp32 top-1 predictions match the original PyTorch model. |
| `android_snippet_for_device_latency_tests.txt` | ML Kit snippet used for the on-device latency/RAM/battery measurements. |

### `results_bundle_final/`

Training and deployment outputs, including:

- **Metrics** — `preprocessing_comparison.csv` (3-seed accuracy, macro/weighted-F1) and
  `preprocessing_significance.csv` (paired t-test / Wilcoxon deltas), plus per-run JSONs such as
  `results_baseline_s42.json` recording the split, seed, and class count.
- **Deployable models** — six metadata-embedded LiteRT models (3 architectures × {fp16, fp32}),
  all from the `baseline`, seed-42 configuration.
- **`labels.txt`** — the shared 2,721-class label file used by every model.

> Adjust this list to match the exact contents of your archived bundle before publishing.

---

## Dataset

Models are trained on **Viet Medi Species 2026**, a GBIF-derived collection of medicinal plant
imagery. For this study the corpus is filtered to classes with **≥ 25 images**, giving:

- **2,721 classes**
- Split (seed 42): **train 210,327 / val 46,419 / test 46,419** (val_frac = test_frac = 0.15)

The dataset is distributed separately under its own DOI (see the badge above). This repository does
**not** redistribute the images; scripts expect them in an ImageFolder-style directory of class
subfolders.

---

## Models

| Backbone | Params | fp32 / fp16 size | Baseline top-1 / top-5 / top-10 |
|----------|-------:|-----------------:|--------------------------------:|
| EfficientFormerV2-S0 | 4,209,490 | 17.31 / 8.72 MB | 68.95% / 85.22% / 89.13% |
| MobileNetV2-100 | 5,709,473 | 22.79 / 11.42 MB | 68.31% / 84.33% / 88.33% |
| MobileViT-XXS | 1,824,465 | 7.64 / 2.35 MB | 57.65% / 77.75% / 83.58% |

Accuracy is the 3-seed mean on the held-out test split under the `baseline` pipeline. A ViT-B/16
reference establishes the accuracy ceiling (baseline top-1 = **84.67%**).

**Preprocessing pipelines evaluated:** `baseline`, `clahe`, `sobel`, `clahe_sobel`. Across all
three backbones, edge-only (`sobel`) preprocessing degrades accuracy sharply, while `clahe` and
`clahe_sobel` produce small, mostly negative deltas relative to `baseline` (see
`preprocessing_significance.csv` for paired tests).

---

## On-device benchmarking

Latency, RAM, cold start, and battery efficiency were measured on **three physical Android
devices** (Xiaomi, Realme, Oppo) via the ML Kit `ImageClassifier` in its **default configuration**,
identical across devices for cross-paper comparability. Full measurements are in
`mobile_benchmarking.csv`.

**Important caveat (see `REPRO_AND_FACTS.md` §10):** ML Kit's default runtime selects its own
backend per device/model, so measured **latency is runtime-dependent rather than raw model
compute** — the same model spans roughly **188 ms (Oppo) to 5,341 ms (Xiaomi)**. Peak RAM is the
clean, comparable cost signal (~280–343 MB across the grid).

Two deployment-path adjustments are baked into the shipped models and were both verified **not** to
change predictions (`parity_check_deployed.py`):

- MobileViT carries a baked-in NHWC transpose.
- EfficientFormerV2 uses an approximated (polynomial) GELU to avoid an unsupported Flex op.

---

## Reproducing the results

The pipeline uses two separate Python environments (exact pins in `REPRO_AND_FACTS.md`).

**1. Training / evaluation** (`phd` env — Python 3.9, `torch==2.6.0+cu124`, `timm==1.0.28`):

```bash
# run the full architecture × preprocessing × seed grid
bash notebooks_n_python_programs/run_grid.sh

# aggregate metrics + significance
python notebooks_n_python_programs/aggregate_preprocessing.py
```

**2. Conversion / deployment** (`aiedge` env — Python 3.11, `ai-edge-litert==2.1.0`,
`tensorflow==2.19.0`):

```bash
# export → NHWC → embed metadata for all six shipped models
bash notebooks_n_python_programs/convert_all.sh

# confirm deployed models match PyTorch predictions
python notebooks_n_python_programs/parity_check_deployed.py
```

On-device numbers are collected by running the classifier from
`android_snippet_for_device_latency_tests.txt` on each phone.

### Environment summary

| Stage | Python | Key packages |
|-------|--------|--------------|
| Training | 3.9.25 | `torch 2.6.0+cu124`, `torchvision 0.21.0`, `timm 1.0.28`, `scikit-learn 1.6.1`, `numpy 2.0.2` |
| Conversion | 3.11.13 | `ai-edge-litert 2.1.0`, `ai-edge-torch 0.7.2`, `litert-torch 0.8.0`, `tensorflow 2.19.0`, `torch 2.6.0+cpu` |

Training hardware: NVIDIA L40 (46 GB), Rocky Linux 9.8.

---

## How to cite

If you use this software or the archived outputs, please cite both the archive and the paper.

```bibtex
@software{tran_mobile_medherb_benchmark_2026,
  author  = {Tran, Trien Phat and Ud Din, Fareed and Brankovic, Ljiljana and Sanin, Cesar and Hester, Susan M.},
  title   = {Mobile Vision-Transformer Benchmark for Fine-Grained Vietnamese Medicinal Plant Classification},
  year    = {2026},
  version = {v1.0.0},
  doi     = {10.XXXX/zenodo.XXXXXXX},
  url     = {https://doi.org/XX.XXXX/zenodo.XXXXXXX}
}
```

<!-- Replace with the final paper citation once available -->
> Paper: Tran, T. P., Ud Din, F., Brankovic, L., Sanin, C., & Hester, S. M. (2026).
> *[Manuscript title]*. [Venue]. DOI: pending.

---

## License

<!-- Choose and confirm before publishing -->
- **Code** (`notebooks_n_python_programs/`): MIT (suggested).
- **Trained models, metrics, and metadata** (`results_bundle_final/`, CSVs): CC BY 4.0 (suggested).
- The underlying imagery is GBIF-derived and governed by the *Viet Medi Species 2026* dataset
  terms; refer to the dataset record for source attributions and reuse conditions.

---

## Acknowledgements

Developed at the University of New England (Armidale, NSW, Australia). Supervisory team: Fareed Ud
Din, Ljiljana Brankovic, Cesar Sanin, and Susan M. Hester. Occurrence imagery derives from GBIF
contributors via the *Viet Medi Species 2026* dataset.
