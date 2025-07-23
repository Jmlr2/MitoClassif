# src/mitoclassif/ _annotator.py

import json
import shutil
from pathlib import Path
from typing import Dict, List, Union


def list_stacks(raw_dir: Union[str, Path]) -> List[Path]:
    """
    Retourne la liste triée des fichiers de piles 3D (.tif, .tiff, .stk)
    présents dans `raw_dir`.
    """
    raw_dir = Path(raw_dir)
    return sorted(
        p
        for p in raw_dir.iterdir()
        if p.suffix.lower() in {".tif", ".tiff", ".stk"}
    )


def save_annotation(
    mapping: Dict[str, str], save_path: Union[str, Path]
) -> None:
    """
    Sauvegarde le dictionnaire d'annotations `{filename: class_label}` dans un fichier JSON.
    `save_path` est le chemin vers le fichier JSON (p. ex. annotations.json).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def load_annotation(save_path: Union[str, Path]) -> Dict[str, str]:
    """
    Charge et retourne un dictionnaire d'annotations depuis un fichier JSON
    créé par `save_annotation`.
    """
    save_path = Path(save_path)
    if not save_path.exists():
        return {}
    with save_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def export_classified_folder(  # (pas utilisé)
    mapping: Dict[str, str],
    raw_dir: Union[str, Path],
    out_dir: Union[str, Path],
) -> None:
    """
    Pour chaque paire (filename → class_label) dans `mapping` :
      - crée `out_dir/class_label/` si nécessaire,
      - copie `raw_dir/filename` dans `out_dir/class_label/filename`.

    `mapping` : dict mapping filename (string) to class label (string).
    `raw_dir`  : dossier contenant les fichiers bruts.
    `out_dir`  : dossier de destination, qui sera créé/modifié.
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    for fname, cls in mapping.items():
        src = raw_dir / fname
        if not src.exists():
            # Ignorer les fichiers manquants
            continue
        dest_dir = out_dir / cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_dir / fname)
