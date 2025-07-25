# src/mitoclass/ _annotator.py

import json
import shutil
from pathlib import Path
from typing import Dict, List, Union


def list_stacks(raw_dir: Union[str, Path]) -> List[Path]:
    """
    Returns the sorted list of 3D stack files (.tif, .tiff, .stk)
    present in `raw_dir`.
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
    Saves the annotation dictionary `{filename: class_label}` to a JSON file.
    `save_path` is the path to the JSON file (e.g., annotations.json).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def load_annotation(save_path: Union[str, Path]) -> Dict[str, str]:
    """
    Loads and returns a dictionary of annotations from a JSON file
    created by `save_annotation`.
    """
    save_path = Path(save_path)
    if not save_path.exists():
        return {}
    with save_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def export_classified_folder(  # (not used)
    mapping: Dict[str, str],
    raw_dir: Union[str, Path],
    out_dir: Union[str, Path],
) -> None:
    """
    For each pair (filename → class_label) in `mapping`:
    - create `out_dir/class_label/` if necessary,
    - copy `raw_dir/filename` to `out_dir/class_label/filename`.

    `mapping`: dict mapping filename (string) to class label (string).
    `raw_dir`: folder containing the raw files.
    `out_dir`: destination folder, which will be created/modified.
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    for fname, cls in mapping.items():
        src = raw_dir / fname
        if not src.exists():
            # Ignore missing files
            continue
        dest_dir = out_dir / cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_dir / fname)
