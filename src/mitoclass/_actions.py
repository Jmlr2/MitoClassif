# src/mitoclassif/_actions.py

from __future__ import annotations

import contextlib
from pathlib import Path

# from typing import Any, Dict, Tuple
from typing import Any

import numpy as np
from napari import Viewer

from ._processor import (
    compute_statistics,
    infer_array,
    load_model,
    make_overlay_rgb,
)

CLASS_LABELS: dict[int, str] = {
    0: "Fond",
    1: "Connecté",
    2: "Fragmenté",
    3: "Intermédiaire",
}

_MODEL_CACHE: Dict[Path, Any] = {}


def _get_model(model_path: Path):
    model_path = Path(model_path)
    mdl = _MODEL_CACHE.get(model_path)
    if mdl is None:
        print(f"[mitoclassif] Chargement modèle : {model_path}")
        mdl = load_model(model_path)
        _MODEL_CACHE[model_path] = mdl
    return mdl


def infer_selected_layer(
    viewer: Viewer,
    *,
    layer=None,
    model_path: Path | None = None,
    patch_size: Tuple[int, int] = (512, 512),
    overlap: Tuple[int, int] = (32, 32),
    batch_size: int = 128,
    to_8bit: bool = False,
    add_table: bool = False,
    **_ignore: Any,
):
    """Infère Mitoclassif sur la couche active (ou fournie)."""

    # --- récupérer la couche active ---
    if layer is None:
        layer = viewer.layers.selection.active
    elif isinstance(layer, str):
        layer = viewer.layers[layer]
    if layer is None:
        print("[mitoclassif] Aucune couche sélectionnée.")
        return None

    # --- charger le modèle ---
    if model_path is None:
        print("[mitoclassif] Aucun modèle fourni (model_path).")
        return None
    model = _get_model(Path(model_path))

    # --- préparer les données & prédire ---
    data = np.asarray(layer.data)
    try:
        best_class = infer_array(
            data=data,
            model=model,
            patch_size=patch_size,
            overlap=overlap,
            batch_size=batch_size,
            to_8bit=to_8bit,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[mitoclass] ERREUR pendant l'inférence : {e}")
        return None

    # ——————————————  PAS DE LABELS  ——————————————
    # On n'appelle plus `viewer.add_labels(...)`.

    # --- statistiques ---
    proportions, global_class = compute_statistics(best_class)
    counts = {c: int((best_class == c).sum()) for c in (1, 2, 3)}
    total_fg = int((best_class > 0).sum())
    cls_name = CLASS_LABELS.get(global_class, str(global_class))

    # --- stocker dans metadata de la couche d’origine ---
    layer.metadata["mitoclassif_proportions"] = proportions
    layer.metadata["mitoclassif_counts"] = counts
    layer.metadata["mitoclassif_total_fg"] = total_fg
    layer.metadata["mitoclassif_global_class"] = int(global_class)

    # --- console ---
    print("\n[mitoclassif] Résultats d'inférence")
    print(f"  Couche source    : {layer.name}")
    print(
        f"  Classe dominante : {cls_name} ({proportions[global_class]:.1f}%)"
    )
    print(f"  Total pixels FG  : {total_fg}")
    print("  Détail par classe :")
    print(f"    Connecté      : {proportions[1]:5.1f}%  ({counts[1]})")
    print(f"    Fragmenté     : {proportions[2]:5.1f}%  ({counts[2]})")
    print(f"    Intermédiaire : {proportions[3]:5.1f}%  ({counts[3]})")

    # --- status bar Napari ---
    with contextlib.suppress(Exception):
        viewer.window._qt_window.statusBar().showMessage(
            f"Mitoclassif: {cls_name} "
            f"(Conn {proportions[1]:.0f}%, Frag {proportions[2]:.0f}%, "
            f"Int {proportions[3]:.0f}%)",
            msecs=6000,
        )

    # ————————————— ALWAYS CREATE OVERLAY —————————————
    overlay_img = make_overlay_rgb(data, best_class)
    o_name = f"{layer.name}_MitoOverlay"
    if o_name in viewer.layers:
        viewer.layers[o_name].data = overlay_img
    else:
        o_layer = viewer.add_image(overlay_img, name=o_name, rgb=True)
        # masquer la couche brute pour plus de lisibilité
        layer.visible = False
        # placer l’overlay sous les autres layers
        src_index = viewer.layers.index(o_layer)
        dest_index = len(viewer.layers) - 1
        viewer.layers.move(src_index, dest_index)

    # --- texte overlay direct ---
    try:
        summary_txt = (
            f"Mitoclassif - {cls_name}\n"
            f"Conn: {proportions[1]:.1f}%  Frag: {proportions[2]:.1f}%  "
            f"Int: {proportions[3]:.1f}%\n"
            f"Pixels FG: {total_fg}"
        )
        viewer.text_overlay.visible = True
        viewer.text_overlay.text = summary_txt
        viewer.text_overlay.location = "top_left"
        viewer.text_overlay.font_size = 12
    except Exception:  # noqa: BLE001
        pass

    # --- table optionnelle ---
    if add_table:
        try:
            import pandas as pd

            df = pd.DataFrame(
                {
                    "layer": [layer.name],
                    "pct_connected": [proportions[1]],
                    "pct_fragmented": [proportions[2]],
                    "pct_intermediate": [proportions[3]],
                    "global_class": [global_class],
                    "total_fg": [total_fg],
                }
            )
            layer.metadata["mitoclassif_table"] = df
        except Exception as e:  # noqa: BLE001e:
            print(f"[mitoclassif] Impossible de créer la table pandas : {e}")

    return {
        "proportions": proportions,
        "global_class": int(global_class),
        "counts": counts,
        "total_fg": total_fg,
        "layer_name": layer.name,
    }
