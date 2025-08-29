
# <img src="https://raw.githubusercontent.com/Jmlr2/MitoClassif/main/assets/mitoclass.png" alt="MitoClass logo" height="60" style="vertical-align: middle;"> Mitoclass

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/mitoclass.svg)](https://pypi.org/project/mitoclass/)
[![Python ≥ 3.10](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)]()
[![napari‑hub](https://img.shields.io/badge/napari--hub-mitoclass-orange.svg)](https://github.com/napari/napari-hub)

<p align="left">
  <img src="https://raw.githubusercontent.com/Jmlr2/MitoClassif/main/assets/imhorphen.png" alt="IMHORPHEN" height="70" style="margin: 0 20px;">
  <img src="https://raw.githubusercontent.com/Jmlr2/MitoClassif/main/assets/LARIS.png" alt="LARIS" height="70" style="margin: 0 20px;">
  <img src="https://raw.githubusercontent.com/Jmlr2/MitoClassif/main/assets/ua.png" alt="Université d'Angers" height="70" style="margin: 0 20px;">
</p>

---

## 1&nbsp;&nbsp;Overview

**Mitoclass** is a *napari* plugin for the qualitative assessment of mitochondrial network morphology.
Inference is **patch‑wise**: each 2‑D patch—obtained from a maximum‑intensity projection of 3‑D stacks—is classified as **connected**, **fragmented**, or **intermediate**.

---

## 2&nbsp;&nbsp;Key features

| Module | Description |
|--------|-------------|
| Patch‑based inference | Analyse an image folder *or* the active napari layer. |
| RGBA heatmaps | Overlay prediction maps as semi‑transparent layers in napari. |
| Global statistics | Compute the proportion of pixels assigned to each morphology and identify the dominant class. |
| 3‑D graph | Interactive Plotly scatter plot of connected / fragmented / intermediate proportions per image. |

---

## 3&nbsp;&nbsp;Requirements

* **Python** ≥ 3.10
* **OS** : Windows, Linux or macOS
* **Hardware** : CPU is sufficient; GPU (CUDA 11+) is recommended for large datasets

---

## 4&nbsp;&nbsp;Installation

### 4.1  PyPI

```bash
pip install mitoclass
```

### 4.2  Reproducible *conda* environment

```bash
conda create -n mitoclass python=3.10
conda activate mitoclass

# (Optional) GPU acceleration
conda install -c conda-forge cudnn=8.9 cuda11.8 tensorflow

pip install mitoclass
```

*Apple Silicon*: install `tensorflow-macos`.

### 4.3  Pre‑trained model

Download the model (`.h5`) from
<https://github.com/Jmlr2/MitoClassif/releases>

---

## 5&nbsp;&nbsp;Usage

### 5.1  Graphical interface

```bash
napari
```

1. Open **Plugins → Mitoclass**.
2. Select the four required paths:

   | Field | Purpose |
   |-------|---------|
   | **Input dir** | Folder of images to analyse (`.tif`, `.tiff`, `.stk`, `.png`). |
   | **Output dir** | Destination folder for CSV and graph files. |
   | **Heatmaps dir** | Folder where heatmaps (`*_map.tif`) will be written. |
   | **Model file** | Pre‑trained Keras model (`.h5`). |

3. Click **Run inference**. A progress bar tracks the number of processed images.
4. After completion:
   * **Show heatmaps** adds the newly generated `*_map.tif` layers to napari.
   * **Show 3D graph** opens `graph3d.html`, displaying the connected/fragmented/intermediate proportions.

*Tip*: Without an *Input dir* you may run **Infer active layer**; results are still saved to *Output dir* and *Heatmaps dir*.

### 5.2  Output structure

| Folder | File(s) | Content |
|--------|---------|---------|
| **Output dir** | `predictions.csv` | Pixel proportion for each class (*connected*, *fragmented*, *intermediate*) and the dominant morphology, one line per image. |
|                | `graph3d.html` | Interactive 3‑D Plotly graph of class proportions. |
| **Heatmaps dir** | `*_map.tif` | One RGBA heatmap per image, ready to overlay in napari. |

---

## 6&nbsp;&nbsp;Licence

This project is released under the **GNU GPL v3** licence.
See the [LICENSE](LICENSE) file for details.
