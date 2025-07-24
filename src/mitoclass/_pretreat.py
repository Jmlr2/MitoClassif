# src/mitoclassif/_pretreat.py

"""
Preprocessing pipeline for microscopy image stacks:
- Reads 3D stacks organized by class in subfolders
- Splits data stratified into train/val/test
- Computes max-intensity projection (MIP)
- Converts to 8-bit or 16-bit as chosen
- Segments via Otsu threshold (on temporary 8-bit copy)
- Extracts patches with overlap
- Labels patches as class or background based on minimum mask pixels
- Saves patches in TIFF format under split/class and produces a manifest CSV
"""
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from skimage.filters import threshold_otsu
from sklearn.model_selection import train_test_split


def max_intensity_projection(data: np.ndarray) -> np.ndarray:
    """Retourne la projection d'intensité maximale le long de l'axe Z."""
    return data.max(axis=0) if data.ndim > 2 else data


def convert_to_8bit(img: np.ndarray) -> np.ndarray:
    """Normalise l'image sur [0,255] et convertit en uint8 pour segmentation."""
    img_f = img.astype(np.float32)
    mn, mx = img_f.min(), img_f.max()
    if mx > mn:
        norm = (img_f - mn) / (mx - mn) * 255.0
    else:
        norm = np.zeros_like(img_f)
    return norm.astype(np.uint8)


def get_patch_positions(length: int, size: int, overlap: int) -> list[int]:
    step = size - overlap
    if size >= length:
        return [0]
    positions = list(range(0, length - size + 1, step))
    last = length - size
    if positions[-1] != last:
        positions.append(last)
    return positions


def extract_patches(
    img: np.ndarray, patch_size: tuple[int, int], overlap: tuple[int, int]
) -> tuple[int, int, np.ndarray]:
    """Générateur de patches (x, y, patch) sur l'image 2D avec recouvrement."""
    h, w = img.shape
    ph, pw = patch_size
    oh, ow = overlap
    ys = get_patch_positions(h, ph, oh)
    xs = get_patch_positions(w, pw, ow)
    for y in ys:
        for x in xs:
            yield x, y, img[y : y + ph, x : x + pw]


def preprocess(
    input_dir: Path,
    output_dir: Path,
    splits: tuple[float, float, float],
    patch_size: tuple[int, int],
    overlap: tuple[int, int],
    min_mask_pixels: int,
    to_8bit: bool = False,
    seed: int = 42,
):
    """
    Exécute la pipeline de prétraitement.

    Args:
        input_dir: dossier racine contenant un sous-dossier par classe.
        output_dir: dossier de sortie pour les patches.
        splits: fractions pour train/val/test (doivent sommer à 1).
        patch_size: taille des patches (hauteur, largeur).
        overlap: recouvrement entre patches (hauteur, largeur).
        min_mask_pixels: nb. minimal de pixels foreground pour conserver la classe.
        to_8bit: si True, convertit et sauvegarde les patches en uint8, sinon en uint16.
        seed: graine pour la reproductibilité du split.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    # 1. Classes
    class_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    class_names = sorted([d.name for d in class_dirs])

    # 2. Collecte
    image_paths: list[Path] = []
    labels: list[str] = []
    for cls in class_names:
        for img in (input_dir / cls).iterdir():
            if img.suffix.lower() in {".tif", ".tiff", ".stk"}:
                image_paths.append(img)
                labels.append(cls)

    # 3. Stratified split
    class_to_idx = {n: i for i, n in enumerate(class_names)}
    y = [class_to_idx[label] for label in labels]  # E741 -> clair
    p_train, p_val, p_test = splits
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, y, train_size=p_train, stratify=y, random_state=seed
    )
    val_ratio = p_val / (p_val + p_test)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        train_size=val_ratio,
        stratify=y_temp,
        random_state=seed,
    )
    split_data = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }

    # 4. Arborescence
    for sp in split_data:
        for cls in class_names + ["background"]:
            (output_dir / sp / cls).mkdir(parents=True, exist_ok=True)

    # 5. Patch extraction avec normalisation globale de la MIP
    manifest = []
    for sp, (X_sp, y_sp) in split_data.items():
        for img_path, lbl_idx in zip(X_sp, y_sp, strict=False):
            data = tiff.imread(img_path)
            mip = max_intensity_projection(data)

            # 5.1 segmentation sur une image 8-bits temporaire
            temp8 = convert_to_8bit(mip)
            try:
                mask = temp8 > threshold_otsu(temp8)
            except Exception:  # noqa: E722
                mask = temp8 > temp8.mean()

            # 5.2 normalisation globale de la MIP
            if to_8bit:
                proc = temp8  # déjà dans [0,255] uint8
            else:
                pf = mip.astype(np.float32)
                mn, mx = pf.min(), pf.max()
                if mx > mn:
                    proc = ((pf - mn) / (mx - mn) * 65535).astype(np.uint16)
                else:
                    proc = np.zeros_like(pf, dtype=np.uint16)

            # 5.3 découpage et sauvegarde des patches
            for x, y, patch in extract_patches(proc, patch_size, overlap):
                n_fg = int(
                    mask[y : y + patch_size[0], x : x + patch_size[1]].sum()
                )
                out_label = (
                    class_names[lbl_idx]
                    if n_fg >= min_mask_pixels
                    else "background"
                )

                fn = f"{img_path.stem}_x{x}_y{y}.tif"
                dest = output_dir / sp / out_label / fn
                tiff.imwrite(str(dest), patch)  # patch déjà en uint8 ou uint16
                manifest.append(
                    {
                        "split": sp,
                        "original": img_path.name,
                        "x": x,
                        "y": y,
                        "label": out_label,
                        "patch_path": str(dest),
                    }
                )

    # 6. Manifest
    # pd.DataFrame(manifest).to_csv(output_dir/'manifest.csv', index=False)
    pd.DataFrame(manifest).to_csv(
        output_dir / "manifest.csv",
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        index=False,
    )
    print(f"Preprocessing complete, manifest: {output_dir/'manifest.csv'}")
