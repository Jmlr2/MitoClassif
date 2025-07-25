# <img src="assets/mitoclass.png" alt="MitoClass logo" height="60" style="vertical-align: middle;"> Mitoclass         



[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/mitoclass.svg)](https://pypi.org/project/mitoclass/)
[![Python ≥ 3.10](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)]()
[![napari‑hub](https://img.shields.io/badge/napari--hub-mitoclass-orange.svg)](https://github.com/napari/napari-hub)

<p align="left">
  <img src="assets/imhorphen.png" alt="IMHORPHEN" height="70" style="margin: 0 20px;">
  <img src="assets/LARIS.png" alt="LARIS" height="70" style="margin: 0 20px;">
  <img src="assets/ua.png" alt="Université d'Angers" height="70" style="margin: 0 20px;">
</p>

A **napari** plugin for the **qualitative** assessment of mitochondrial network morphology.

---

## 1. Overview

Mitoclass provides an end‑to‑end workflow to classify mitochondrial morphology—**connected**, **fragmented**, or **intermediate**—directly inside the napari viewer.
Beyond inference, the plugin offers tools for :

- data annotation;
- model training and improvement;
- interactive result visualisation.

All functionality is accessible through a graphical user interface.

---

## 2. Key features

| Module            | Functionality                                                                                                                                                                       |
|-------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Prediction**    | Pixel‑wise classification of 3‑D stacks (automatic maximum‑intensity projection). Batch processing, class overlays, per‑image statistics, CSV summary.                              |
| **Annotation**    | Fast image labelling to standardise classes for model training.                                                                                                                     |
| **Pre‑processing**| Generation of normalised patches with stratified *train/val/test* split.                                                                                                            |
| **Training**      | Train a CNN or **fine‑tune** existing models.                                                                                                                                       |
| **Visualisation** | Interactive heatmaps and 3‑D scatter plots (Plotly) of class proportions.                                                                                                           |

---

## 3. Requirements

- **Python** ≥ 3.10
- **Operating systems**: Linux, macOS, or Windows
- **Hardware**: CPU supported; GPU (CUDA 11+) recommended for large‑scale inference and training

---

## 4. Installation

### 4.1. Stable release (PyPI)

```bash
pip install mitoclass
```

> Installs the plugin and its dependencies (napari, Qt, etc.).

### 4.2. Reproducible conda environment

```bash
conda create -n mitoclass python=3.10
conda activate mitoclass

# (Optional) TensorFlow GPU (e.g. Linux CUDA 11.8)
conda install -c conda-forge cudnn=8.9 cuda11.8 tensorflow

pip install mitoclass
```

💡 **Apple Silicon**: use `tensorflow-macos` instead.

### 4.3. Download the pre‑trained model

<https://github.com/Jmlr2/MitoClassif/releases>

---

## 5. Usage

### 5.1. Graphical interface

```bash
napari
```

1. Open the **Mitoclass** widget from the **Plugins** menu.
2. Select the desired tab.

---

### 5.2. Annotation

| Item        | Description                                                                                                                                                                      |
|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Goal**    | Create an *image → class* mapping to bootstrap or expand a training dataset.                                                                                                     |
| **Input**   | Folder of unlabelled images (`*.tif`, `*.tiff`, `*.stk`).                                                                                                                         |
| **Output**  | Images copied or moved to `annot_output/<ClassName>/`, sorted into one folder per class.                                                                                          |

---

### 5.3. Pre‑processing

**Goal**: convert 3‑D stacks into normalised 2‑D patches for CNN training.

**Input structure**:

```
raw_input/
├── Connected/
├── Fragmented/
└── Intermediate/
```

**Steps**:

1. Maximum‑intensity projection (MIP)
2. Intensity normalisation (8‑bit or 16‑bit)
3. Otsu segmentation
4. Patch extraction (configurable size/overlap)
5. Patch labelling (class vs. background)
6. Stratified *train/val/test* split

**Output structure**:

```
pp_output/
├── train/
│   ├── Connected/
│   ├── Fragmented/
│   ├── Intermediate/
│   └── background/
├── val/
├── test/
└── manifest.csv
```

**CSV columns**: `split`, `original`, `x`, `y`, `label`, `patch_path`

---

### 5.4. Training

| Item          | Description                                                                                                                                                                  |
|---------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Goal**      | Train a CNN (or **fine‑tune** an existing model) on the patches.                                                                                                              |
| **Input**     | Patch folders (`train/`, `val/`, `test/`).                                                                                                                                    |
| **Parameters**| Patch size, batch size, epochs, learning rate, patience, bit depth, pre‑trained model (`.h5`).                                                                               |
| **Outputs**   | In the output directory: `best_model.h5`, `best_model_history.csv`, `best_model_test_metrics.csv`, `best_model_classification_report.txt`.                                    |

---

### 5.5. Prediction / Inference

| Item          | Description                                                                                                                                                                              |
|---------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Goal**      | Classify new images/stacks and compute class proportions.                                                                                                                                |
| **Inputs**    | - Folder of images (`.tif`, `.tiff`, `.stk`, `.png`) ; <br>- Active layer in napari                                                                                                      |
| **Parameters**| Patch size, overlap, batch size, bit‑depth conversion, model path (`.h5`).                                                                                                               |
| **Outputs**   | - `predictions.csv` (image, % connected, % fragmented, % intermediate, *global_class*) ; <br>- Folder of heatmaps (`*_map.tif`) ; <br>- Optional global summary CSV                     |
| **Interactive**| Overlay in napari and interactive 3‑D scatter plot (`graph3d.html`).                                                                                                                    |

---

## 6. Licence

This software is released under the **GNU GPL v3** licence.
Refer to the [LICENSE](LICENSE) file for details.
