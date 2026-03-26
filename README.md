# URS-10T Urinalysis Strip Detection System

**Team:** Kuwait Tech Titans  
**Members:** Manar Almutairi · Mona Alajmi · Nourah AlZahmoul
**Platform:** Huawei ModelArts · MindSpore 1.7.0 · Huawei OBS  
**Version:** v1.0 — Trained 2026-03-08

---

## Overview

An end-to-end AI pipeline that automates reading of 10-pad URS-10T urine reagent test strips using convolutional neural networks on Huawei Cloud. A photo of the strip is taken, each of the 10 reagent pads is located and extracted, and a trained MindSpore CNN classifies each pad's colour against clinical reference values — producing a structured severity report in under 200 ms.

**What it detects:**

| Pad | Biomarker | Clinical Significance |
|-----|-----------|----------------------|
| 1 | Leukocytes | White blood cells — UTI indicator |
| 2 | Nitrite | Bacterial infection |
| 3 | Urobilinogen | Liver function |
| 4 | Protein | Kidney disease / preeclampsia |
| 5 | pH | Acid-base balance |
| 6 | Blood | Kidney injury / haematuria |
| 7 | Specific Gravity | Hydration / kidney concentration |
| 8 | Ketone | Diabetic ketoacidosis |
| 9 | Bilirubin | Liver / bile duct disease |
| 10 | Glucose | Diabetes screening |

---

## Requirements

### Huawei Cloud Setup

| Resource | Configuration |
|----------|--------------|
| ModelArts Notebook | Image: `mindspore_1_7_0` · Flavor: CPU 2vCPUs 8GB |
| OBS Bucket | Name: `urinalysis-data` · Region: AF-Johannesburg |
| Access Keys | AK/SK from: My Credentials → Access Keys |

### Python Dependencies

```
mindspore==1.7.0
opencv-python-headless>=4.5.5
numpy>=1.21.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
esdk-obs-python>=3.21.4
ipywidgets>=7.6.5
Pillow>=8.0.0
```

Install inside the notebook (Cell 1 does this automatically):

```bash
pip install opencv-python-headless matplotlib scikit-learn esdk-obs-python ipywidgets --quiet
```

> **Note:** `mindspore` is pre-installed in the ModelArts `mindspore_1_7_0` image. Do not reinstall it.

---

## Notebook Structure

The main notebook is `urinalysis_huawei_full.ipynb`. Run cells **in order**.

| Cell | Name | What it does |
|------|------|-------------|
| Cell 1 | Setup & OBS Config | Installs packages, defines OBS credentials and helper functions (`obs_upload`, `obs_download`) |
| Cell 2 | Definitions | All functions: strip detection, CNN architecture (`PadClassifier`), training loop, inference, visualisation |
| Cell 3 | Train / Load Models | Checks OBS for existing checkpoints → loads them. If not found → trains 10 CNNs and uploads to OBS |
| Cell 3b | Evaluation Metrics | Per-pad accuracy, F1, ROC-AUC, confusion matrices, ROC curves — all saved to OBS `results/` |
| Cell 4 | Scan from OBS | Downloads latest image from OBS `images/` folder and runs full inference |
| Cell 5 | Upload & Scan | ipywidgets file upload button → runs inference on uploaded photo |
| Cell 6 | Demo Mode | Runs inference on a built-in synthetic scenario (`uti`, `healthy`, `diabetes`, `kidney`) |

---

## Quick Start

### Step 1 — Configure OBS credentials

Open **Cell 1** and fill in your keys:

```python
OBS_CONFIG = {
    "access_key": "YOUR_AK_HERE",
    "secret_key": "YOUR_SK_HERE",
    "endpoint":   "https://obs.af-south-1.myhuaweicloud.com",
    "bucket":     "urinalysis-data",
}
```

### Step 2 — Run all cells in order

```
Cell 1 → Cell 2 → Cell 3 → Cell 3b → Cell 6 (demo)
```

### Step 3 — Run inference

**Option A — Demo (no photo needed):**
```python
# In Cell 6, set the scenario:
scenario = 'uti'        # UTI with critical leukocytes + blood
# scenario = 'healthy'  # All pads normal
# scenario = 'diabetes' # Elevated glucose + ketones
# scenario = 'kidney'   # Proteinuria + blood + specific gravity abnormal
```

**Option B — Upload your own photo:**
Run Cell 5 → click the upload button → select a strip photo

**Option C — Scan from OBS:**
Upload your photo to `obs://urinalysis-data/images/` then run Cell 4

---

## Training

Cell 3 handles training automatically. On first run it:

1. Generates 3,240 synthetic 32×32 images from the URS-10T colour reference chart
2. Trains 10 independent `PadClassifier` CNNs (one per biomarker)
3. Saves the best checkpoint per pad to OBS `models/{pad}.ckpt`

On subsequent runs it detects existing checkpoints and skips retraining (`[OBS] skip`).

**To force retrain:** delete the `.ckpt` files from OBS `models/` folder, then re-run Cell 3.

**Training configuration** (editable in Cell 2):

```python
N_SAMPLES  = 60     # images generated per colour level
EPOCHS     = 40     # max training epochs per model
PATIENCE   = 8      # early stopping patience
BATCH_SIZE = 32
IMG_SIZE   = 32     # CNN input size (32×32 pixels)
LR         = 1e-3   # Adam learning rate
```

---

## CNN Architecture

One `PadClassifier` is trained per biomarker pad.

```
Input: (B, 3, 32, 32)  — NCHW format (MindSpore convention)
    ↓
Conv2d(3→32, k=3) + BatchNorm2d + ReLU + MaxPool2d(2)
    ↓  (B, 32, 16, 16)
Conv2d(32→64, k=3) + ReLU + AvgPool2d(kernel=16)
    ↓  (B, 64, 1, 1)
Flatten → (B, 64)
    ↓
Dropout(keep_prob=0.7) + Dense(64→64) + ReLU
    ↓
Dense(64→n_classes) + Softmax
```

| Property | Value |
|----------|-------|
| Parameters | ~50,000 per model |
| Format | MindSpore `.ckpt` |
| All 10 models | ~520 KB total |
| Inference | 1.9 ms (CNN only) · <200 ms (full pipeline) |

> **MindSpore 1.7 note:** `nn.AdaptiveAvgPool2d` does not exist in 1.7.
> This project uses `nn.AvgPool2d(kernel_size=16)` instead — identical result on a 16×16 feature map.

---

## Evaluation Results

Measured on 20% held-out validation split — actual results from ModelArts run 2026-03-08:

| Pad | ACC | F1-W | ROC-AUC |
|-----|-----|------|---------|
| Leukocytes | 0.483 | 0.382 | 0.891 |
| Nitrite | 0.667 | 0.533 | 1.000 |
| Urobilinogen | 0.717 | 0.711 | 0.937 |
| Protein | 0.597 | 0.507 | 0.954 |
| pH | 0.714 | 0.711 | 0.968 |
| Blood | 0.736 | 0.720 | 0.956 |
| Specific Gravity | 0.440 | 0.391 | 0.841 |
| Ketone | 0.639 | 0.609 | 0.939 |
| Bilirubin | **0.938** | **0.938** | **0.996** |
| Glucose | 0.722 | 0.732 | 0.966 |
| **MEAN** | **0.665** | **0.623** | **0.945** |

Mean ROC-AUC 0.945 confirms the CNNs correctly learned colour gradients for all 10 biomarkers.

---

## OBS Bucket Structure

```
obs://urinalysis-data/
├── training_data/
│   ├── urinalysis_dataset.csv      ← 3,240-row feature dataset
│   ├── urinalysis_pad_labels.csv   ← colour reference table
│   └── label_map.json
├── models/
│   ├── Leukocytes.ckpt
│   ├── Nitrite.ckpt
│   ├── Urobilinogen.ckpt
│   ├── Protein.ckpt
│   ├── pH.ckpt
│   ├── Blood.ckpt
│   ├── Specific Gravity.ckpt
│   ├── Ketone.ckpt
│   ├── Bilirubin.ckpt
│   ├── Glucose.ckpt
│   └── label_map.json
├── images/          ← upload strip photos here for Cell 4
└── results/
    ├── confusion_matrices.png
    ├── roc_curves.png
    ├── metrics_summary.png
    └── last_result.png
```

---

## Severity Reference

| Level | Colour | Meaning |
|-------|--------|---------|
| 0 — Normal | 🟢 Green | Within reference range |
| 1 — Mild | 🔵 Blue | Slightly outside range — monitor |
| 2 — Moderate | 🟠 Orange | Consult a doctor |
| 3 — Critical | 🔴 Red | Visit a doctor today |

Overall verdict = highest severity pad detected.

---

> ⚠️ **Disclaimer:** This is a screening tool only — not a substitute for professional medical diagnosis.

---

*URS-10T Urinalysis Detector · Kuwait Tech Titans · Huawei ModelArts · MindSpore 1.7.0*
