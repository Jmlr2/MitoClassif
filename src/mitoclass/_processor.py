# src/mitoclassif/_processor.py

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf
import tifffile as tiff

from ._pretreat import (
    convert_to_8bit,
    extract_patches,
)


def aggregate_pixelwise(
    classes: np.ndarray,
    scores: np.ndarray,
    coords: list[tuple[int, int]],
    img_shape: tuple[int, int],
    patch_size: tuple[int, int],
) -> np.ndarray:
    """Agrège pixel par pixel la meilleure classe selon le score."""
    ph, pw = patch_size
    H, W = img_shape
    best_score = np.full((H, W), -np.inf, dtype=np.float32)
    best_class = np.zeros((H, W), dtype=np.uint8)
    for cls, sc, (x, y) in zip(classes, scores, coords, strict=False):
        region_score = best_score[y : y + ph, x : x + pw]
        mask = sc > region_score
        best_score[y : y + ph, x : x + pw][mask] = sc
        best_class[y : y + ph, x : x + pw][mask] = cls
    return best_class


def compute_statistics(best_class: np.ndarray) -> tuple[dict[int, float], int]:
    """Calcule proportions et global_class d'une carte de classes."""
    mask = best_class > 0
    total = int(mask.sum())
    proportions: dict[int, float] = {}
    counts: dict[int, int] = {}
    for c in (1, 2, 3):
        cnt = int((best_class == c).sum())
        counts[c] = cnt
        proportions[c] = (cnt / total * 100) if total > 0 else 0.0
    global_class = max(counts, key=counts.get) if total > 0 else 0
    return proportions, global_class


# ——————————— Méthode de prédiction ——————————— #


# --- NEW: core infer function that works on an in-memory array -----------
def infer_array(
    data: np.ndarray,
    model: tf.keras.Model,
    patch_size: tuple[int, int],
    overlap: tuple[int, int],
    batch_size: int = 128,
    to_8bit: bool = False,
) -> np.ndarray:
    """
    Infère la classification d'un tableau en mémoire (2D ou 3D stack).

    Parameters
    ----------
    data : np.ndarray
        Image 2D (Y, X) ou stack 3D (Z, Y, X) ou plus large (napari peut contenir
        canaux, temps...). Seuls les deux DERNIERS axes seront pris comme image.
    model : tf.keras.Model
        Modèle Keras chargé.
    patch_size, overlap, batch_size, to_8bit
        Idem que `infer_image`.

    Returns
    -------
    best_class : np.ndarray (uint8, shape=(Y, X))
        Carte de classes agrégée pixelwise.
    """
    # --- squeeze to 2D (MIP sur la profondeur si >2D) --------------------
    # --- squeeze to 2D ou convertir RGB→gris / MIP si stack multidim. ---
    arr = np.asarray(data)
    # Si image RGB ou RGBA (H, W, 3 ou 4) → convertir en gris
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        # moyenne des 3 canaux couleur
        arr = arr[..., :3].mean(axis=-1)
    # Si stack >2D (e.g. (Z, Y, X) ou (T, Z, Y, X)) → MIP sur tous les axes sauf les 2 derniers
    elif arr.ndim > 2:
        reduce_axes = tuple(range(arr.ndim - 2))
        arr = arr.max(axis=reduce_axes)
    # À présent arr est garanti 2D (Y, X)
    arr2d = arr

    # Normalisation même logique que dans infer_image
    if to_8bit:
        img = convert_to_8bit(arr2d).astype(np.float32) / 255.0
    else:
        print("16 bits")
        pf = arr2d.astype(np.float32)
        mn, mx = pf.min(), pf.max()
        img = (pf - mn) / (mx - mn) if mx > mn else np.zeros_like(pf)

    H, W = img.shape
    coords: list[tuple[int, int]] = []
    patches: list[np.ndarray] = []
    for x, y, patch in extract_patches(img, patch_size, overlap):
        coords.append((x, y))
        patches.append(patch)

    X = np.stack(patches, axis=0).astype(np.float32)
    if X.ndim == 3:  # add channel axis
        X = X[..., np.newaxis]

    bs = min(batch_size, len(X))
    probas = model.predict(X, batch_size=bs, verbose=0)
    classes = np.argmax(probas, axis=1).astype(np.uint8)
    scores = np.max(probas, axis=1).astype(np.float32)

    return aggregate_pixelwise(classes, scores, coords, (H, W), patch_size)


def infer_image(
    path: Path,
    model: tf.keras.Model,
    patch_size: tuple[int, int],
    overlap: tuple[int, int],
    batch_size: int = 128,
    to_8bit: bool = False,
) -> np.ndarray:
    """Version historique qui lit sur disque puis appelle infer_array."""
    data = tiff.imread(path)
    return infer_array(
        data=data,
        model=model,
        patch_size=patch_size,
        overlap=overlap,
        batch_size=batch_size,
        to_8bit=to_8bit,
    )


# ——————————— Chargement du modèle & traitement de dossier ——————————— #


def load_model(model_path: Path) -> tf.keras.Model:
    tf.keras.layers.CustomInputLayer = tf.keras.layers.InputLayer

    class CustomInputLayer(tf.keras.layers.InputLayer):
        def __init__(self, *args, batch_shape=None, **kwargs):
            if batch_shape is not None:
                kwargs["input_shape"] = tuple(batch_shape[1:])
            kwargs.pop("batch_shape", None)
            super().__init__(*args, **kwargs)

    return tf.keras.models.load_model(
        model_path,
        custom_objects={
            "InputLayer": CustomInputLayer,
            "DTypePolicy": tf.keras.mixed_precision.Policy,
        },
        compile=False,
    )


def make_overlay_rgb(
    base_img: np.ndarray,
    class_map: np.ndarray,
    colors: dict[int, tuple[int, int, int]] | None = None,
    alpha_base: float = 0.7,
    alpha_map: float = 0.3,
) -> np.ndarray:
    """
    Construit une image RGB overlay pour affichage, en gérant :
      - stacks multi-axes -> MIP sur tous les axes sauf les deux derniers,
      - images RGB/RGBA -> conversion en gris,
      - dtype non-uint8 -> convert_to_8bit automatique,
      - recadrage si base_img et class_map ont des tailles differentes.
    """
    arr = np.asarray(base_img)

    # 1) RGB/RGBA -> moyenne des 3 canaux
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr2d = arr[..., :3].astype(np.float32).mean(axis=-1)
    # 2) Stack >2D -> MIP sur tous les axes sauf X et Y
    elif arr.ndim > 2:
        reduce_axes = tuple(range(arr.ndim - 2))
        arr2d = arr.max(axis=reduce_axes)
    else:
        arr2d = arr.astype(np.float32)

    # 3) Convertir en uint8 si besoin
    if arr2d.dtype != np.uint8:
        from ._pretreat import convert_to_8bit

        arr2d = convert_to_8bit(arr2d)

    # 4) Empiler en RGB
    base_rgb = np.stack([arr2d] * 3, axis=-1)

    # 4b) Recadrage si mismatch de dimensions
    bh, bw = base_rgb.shape[:2]
    ch, cw = class_map.shape
    if (bh, bw) != (ch, cw):
        H, W = min(bh, ch), min(bw, cw)
        base_rgb = base_rgb[:H, :W]
        class_map = class_map[:H, :W]

    # 5) Couleurs par defaut
    if colors is None:
        colors = {
            1: (255, 0, 0),  # connecté en rouge
            2: (0, 255, 0),  # fragmenté en vert
            3: (0, 0, 255),  # intermédiaire en bleu
        }

    # 6) Construire la carte couleur
    color_map = np.zeros_like(base_rgb)
    for cls, rgb in colors.items():
        color_map[class_map == cls] = rgb

    # 7) Fusion alpha et retour
    return (base_rgb * alpha_base + color_map * alpha_map).astype(np.uint8)


def process_folder(
    input_dir: Path,
    output_dir: Path,
    map_dir: Path,
    model_path: Path,
    patch_size: tuple[int, int] = (512, 512),
    overlap: tuple[int, int] = (32, 32),
    batch_size: int = 128,
    to_8bit: bool = False,
) -> Iterator[Union[int, pd.DataFrame]]:
    """
    Traite un dossier d'images et yield l'indice pour la progression à chaque image.
    En fin de traitement, return le DataFrame des résultats.
    """
    model = load_model(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)

    images = [
        p
        for p in sorted(input_dir.iterdir())
        if p.suffix.lower() in {".tif", ".tiff", ".stk"}
    ]
    results: list[dict] = []

    for idx, img_path in enumerate(images, start=1):
        # inférence
        best_class = infer_image(
            img_path,
            model,
            patch_size,
            overlap,
            batch_size=batch_size,
            to_8bit=to_8bit,
        )
        props, gclass = compute_statistics(best_class)
        results.append(
            {
                "image": img_path.name,
                "pct_connected": props[1],
                "pct_fragmented": props[2],
                "pct_intermediate": props[3],
                "global_class": gclass,
            }
        )

        # overlay
        data = tiff.imread(img_path)
        overlay = make_overlay_rgb(data, best_class)
        tiff.imwrite(map_dir / f"{img_path.stem}_map.tif", overlay)

        # yield l'indice pour la progression
        yield idx

    # fin de boucle → on compose le DataFrame, on l'exporte et on le return
    df = pd.DataFrame(results)

    df.to_csv(
        output_dir / "predictions.csv",
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        index=False,
    )
    return df


def _cli():
    p = argparse.ArgumentParser(description="Inference mitoclassif")
    p.add_argument("--input-dir", "-i", required=True, type=Path)
    p.add_argument("--output-dir", "-o", required=True, type=Path)
    p.add_argument("--map-dir", "-m", required=True, type=Path)
    p.add_argument("--model", "-M", required=True, type=Path)
    p.add_argument(
        "--patch-size",
        "-p",
        nargs=2,
        type=int,
        default=(512, 512),
        help="Taille des patchs: H W",
    )
    p.add_argument(
        "--overlap",
        "-l",
        nargs=2,
        type=int,
        default=(32, 32),
        help="Recouvrement: H W",
    )
    p.add_argument(
        "--to-8bit",
        action="store_true",
        help="Convertir les images en 8 bits avant inférence",
    )
    args = p.parse_args()
    process_folder(
        args.input_dir,
        args.output_dir,
        args.map_dir,
        args.model,
        tuple(args.patch_size),
        tuple(args.overlap),
        to_8bit=args.to_8bit,
    )


if __name__ == "__main__":
    _cli()
