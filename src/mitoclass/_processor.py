# src/mitoclass/_processor.py

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
    """Aggregate pixel by pixel the best class according to the score."""
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
    """Calculate proportions and global_class of a class map."""
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
    Infers the classification of an in-memory array (2D or 3D stack).

    Parameters
    ----------
    data: np.ndarray
    2D image (Y, X) or 3D stack (Z, Y, X) or larger (napari can contain
    channels, time, etc.). Only the LAST two axes will be used as the image.
    model: tf.keras.Model
    Loaded Keras model.
    patch_size, overlap, batch_size, to_8bit
    Same as `infer_image`.

    Returns
    -------
    best_class: np.ndarray (uint8, shape=(Y, X))
    Pixel-wise aggregated class map.
    """
    # --- squeeze to 2D (MIP on depth if >2D) --------------------
    arr = np.asarray(data)
    # f RGB or RGBA (H, W, 3 or 4) image → convert to gray
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        # average of the 3 color channels
        arr = arr[..., :3].mean(axis=-1)
    # If stack >2D (e.g. (Z, Y, X))→ MIP on all axes except the last 2
    elif arr.ndim > 2:
        reduce_axes = tuple(range(arr.ndim - 2))
        arr = arr.max(axis=reduce_axes)
    # Now arr is guaranteed to be 2D (Y, X)
    arr2d = arr

    # Normalization same logic as in infer_image
    if to_8bit:
        img = convert_to_8bit(arr2d).astype(np.float32) / 255.0
    else:
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
    """Version that reads from disk and then calls infer_array."""
    data = tiff.imread(path)
    return infer_array(
        data=data,
        model=model,
        patch_size=patch_size,
        overlap=overlap,
        batch_size=batch_size,
        to_8bit=to_8bit,
    )


# ——————————— Model loading & file processing ——————————— #


def load_model(model_path: Path) -> tf.keras.Model:
    tf.keras.layers.CustomInputLayer = tf.keras.layers.InputLayer

    class CustomInputLayer(tf.keras.layers.InputLayer):
        def __init__(self, *args, batch_shape=None, **kwargs):
            if batch_shape is not None:
                kwargs["shape"] = tuple(batch_shape[1:])
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
    Builds an RGB overlay image for display, handling:
    - multi-axis stacks -> MIP on all axes except the last two,
    - RGB/RGBA images -> conversion to grayscale,
    - non-uint8 dtype -> automatic convert_to_8bit,
    - cropping if base_img and class_map have different sizes.
    """
    print("!!! [DBG] make_overlay_rgb CALLED (CE NE DEVRAIT PLUS ARRIVER)")
    arr = np.asarray(base_img)

    # 1) RGB/RGBA -> average of the 3 channels
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr2d = arr[..., :3].astype(np.float32).mean(axis=-1)
    # 2) Stack >2D -> MIP on all axes except X and Y
    elif arr.ndim > 2:
        reduce_axes = tuple(range(arr.ndim - 2))
        arr2d = arr.max(axis=reduce_axes)
    else:
        arr2d = arr.astype(np.float32)

    # 3) Convert to uint8 if needed
    if arr2d.dtype != np.uint8:

        arr2d = convert_to_8bit(arr2d)

    # 4) Stack in RGB
    base_rgb = np.stack([arr2d] * 3, axis=-1)

    # 4b) Crop if dimension mismatch
    bh, bw = base_rgb.shape[:2]
    ch, cw = class_map.shape
    if (bh, bw) != (ch, cw):
        H, W = min(bh, ch), min(bw, cw)
        base_rgb = base_rgb[:H, :W]
        class_map = class_map[:H, :W]

    # 5) Default colors
    if colors is None:
        colors = {
            1: (255, 0, 0),  # connected in red
            2: (0, 255, 0),  # fragmented in green
            3: (0, 0, 255),  # intermediate in blue
        }

    # 6) Build the color map
    color_map = np.zeros_like(base_rgb)
    for cls, rgb in colors.items():
        color_map[class_map == cls] = rgb

    # 7) Alpha Merge and Return
    return (base_rgb * alpha_base + color_map * alpha_map).astype(np.uint8)


def build_heatmap_rgba(
    class_map: np.ndarray,
    colors: dict[int, tuple[int, int, int]] | None = None,
    alpha: float = 0.4,
) -> np.ndarray:

    print(
        "[DBG] build_heatmap_rgba  → shape:",
        class_map.shape,
        "dtype:",
        class_map.dtype,
        "uniq:",
        np.unique(class_map),
    )
    if colors is None:
        colors = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)}

    h, w = class_map.shape
    heat = np.zeros((h, w, 4), dtype=np.uint8)
    a = int(alpha * 255)
    for cls, (r, g, b) in colors.items():
        mask = class_map == cls
        heat[mask] = (r, g, b, a)
    return heat


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
    model = load_model(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # map_dir : on le recrée proprement (supprime anciens .tif)
    map_dir.mkdir(parents=True, exist_ok=True)
    for old in map_dir.glob("*.tif"):
        old.unlink()

    images = [
        p
        for p in sorted(input_dir.iterdir())
        if p.suffix.lower() in {".tif", ".tiff", ".stk", ".png"}
    ]

    results: list[dict] = []

    for idx, img_path in enumerate(images, start=1):
        # ------------------ inférence ----------------------------
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

        # ------------------ heat‑map RGBA -------------------------
        heatmap_rgba = build_heatmap_rgba(best_class, alpha=0.4)
        assert heatmap_rgba.shape[-1] == 4, "Masque non‑RGBA !?"
        print(
            "[DBG] write",
            f"{img_path.stem}_map.tif",
            "| shape",
            heatmap_rgba.shape,
            "alpha uniq",
            np.unique(heatmap_rgba[..., 3]),
        )

        tiff.imwrite(map_dir / f"{img_path.stem}_map.tif", heatmap_rgba)

        # notifier la progression
        yield idx

    # ------------------ CSV final --------------------------------
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
