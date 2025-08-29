# src/mitoclass/_widget.py

import webbrowser
from pathlib import Path

import pandas as pd
import tensorflow as tf
import tifffile as tiff
from napari.qt.threading import thread_worker
from qtpy.QtCore import QStandardPaths
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from tifffile import imread

from ._processor import (
    build_heatmap_rgba,
    compute_statistics,
    infer_array,
    load_model,
)
from ._utils import status


class MitoclassWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self._status = lambda msg, msecs=10000: status(msg, msecs, self.viewer)
        self.viewer = napari_viewer
        self.df_new = None

        # Master CSV setup
        appdata = QStandardPaths.writableLocation(
            QStandardPaths.AppDataLocation
        )
        self.master_dir = Path(appdata) / "mitoclass"
        self.master_dir.mkdir(parents=True, exist_ok=True)
        self.master_path = self.master_dir / "predictions.csv"
        self.csv_results = None
        if self.master_path.exists():
            try:
                self.csv_results = pd.read_csv(
                    self.master_path,
                    sep=";",
                    decimal=",",
                    encoding="utf-8-sig",
                )
            except Exception as e:
                self._status(f"Error reading master CSV: {e}")
                self.csv_results = None

        self.paths = {
            "input": None,
            "output": None,
            "map": None,
            "model": None,
        }
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()

        # File selectors
        self.input_dir_btn = QPushButton("Choose input folder")
        self.output_dir_btn = QPushButton("Choose output folder")
        self.map_dir_btn = QPushButton("Select heatmaps folder")
        self.model_path_btn = QPushButton("Choose model .h5")
        form.addRow("Input dir:", self.input_dir_btn)
        form.addRow("Output dir:", self.output_dir_btn)
        form.addRow("Map dir:", self.map_dir_btn)
        form.addRow("Model file:", self.model_path_btn)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setValue(16)
        form.addRow("Batch size:", self.batch_spin)

        # Action buttons
        self.add_to_master_cb = QCheckBox("Add to master")
        self.run_btn = QPushButton("Run Inference")
        self.clear_master_btn = QPushButton("Clear master")
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        form.addRow(self.add_to_master_cb)
        form.addRow(self.run_btn)
        form.addRow(self.clear_master_btn)
        form.addRow(self.progress)

        # Napari-layer inference
        self.layer_infer_btn = QPushButton("Infer active layer")
        form.addRow(self.layer_infer_btn)

        # Heatmaps and 3D graph
        self.view_heatmaps_btn = QPushButton("Show heatmaps")
        self.view_heatmaps_btn.setEnabled(False)
        self.graph_source_combo = QComboBox()
        self.graph_source_combo.addItems(["Master Data", "Last Session"])
        self.view_3d_btn = QPushButton("Display 3D graph")
        self.view_3d_btn.setEnabled(False)
        form.addRow(self.view_heatmaps_btn)
        form.addRow("3D graph source:", self.graph_source_combo)
        form.addRow(self.view_3d_btn)

        layout.addLayout(form)

        # Connections
        self.input_dir_btn.clicked.connect(lambda: self._choose_dir("input"))
        self.output_dir_btn.clicked.connect(lambda: self._choose_dir("output"))
        self.map_dir_btn.clicked.connect(lambda: self._choose_dir("map"))
        self.model_path_btn.clicked.connect(self._choose_model)
        self.run_btn.clicked.connect(self._run_inference)
        self.clear_master_btn.clicked.connect(self._clear_master)
        self.layer_infer_btn.clicked.connect(self._infer_active_layer)
        self.view_heatmaps_btn.clicked.connect(self._show_heatmaps)
        self.view_3d_btn.clicked.connect(self._show_3d)

    # ... [rest of methods unchanged] ...

    def _clear_master(self):
        """Remove master CSV and reset history."""
        try:
            if self.master_path.exists():
                self.master_path.unlink()
            self.csv_results = None
            self._status("Master CSV cleared.")
        except Exception as e:
            self._status(f"Error clearing master: {e}")

    def _choose_dir(self, key):
        d = QFileDialog.getExistingDirectory(self, "Select a folder")
        if d:
            self.paths[key] = Path(d)
            getattr(self, f"{key}_dir_btn").setText(f"{key.capitalize()}: {d}")

    def _choose_model(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select a model .h5", filter="*.h5"
        )
        if f:
            self.paths["model"] = Path(f)
            self.model_path_btn.setText(f"Model: {Path(f).name}")
        else:
            self._status("No model selected.")

    def _run_inference(self):
        missing = [
            k
            for k in ("input", "output", "map", "model")
            if self.paths.get(k) is None
        ]
        if missing:
            self._status(f"Missing path(s): {', '.join(missing)}")
            return
        from ._processor import process_folder

        # build image list & progress bar
        images = [
            p
            for p in sorted(self.paths["input"].iterdir())
            if p.suffix.lower() in {".tif", ".tiff", ".stk", ".png"}
        ]
        total = len(images)
        if total == 0:
            return self._status("Input folder contains no images.")

        self.progress.setRange(0, total)
        self.progress.setValue(0)
        self.progress.setVisible(True)

        # fixed defaults
        patch_size = (256, 256)
        overlap = (128, 128)
        batch_size = self.batch_spin.value()
        to_8bit = True

        self.run_btn.setEnabled(False)

        @thread_worker(
            connect={
                "yielded": lambda v: self.progress.setValue(v),
                "returned": lambda df: self._handle_finished(df),
                "errored": lambda e: self._handle_error(e),
            }
        )
        def worker():
            df = yield from process_folder(
                input_dir=self.paths["input"],
                output_dir=self.paths["output"],
                map_dir=self.paths["map"],
                model_path=self.paths["model"],
                patch_size=patch_size,
                overlap=overlap,
                batch_size=batch_size,
                to_8bit=to_8bit,
            )
            return df

        worker()

    def _handle_finished(self, df_new: pd.DataFrame):
        self.df_new = df_new
        if self.add_to_master_cb.isChecked():
            if self.csv_results is not None:
                master = pd.concat(
                    [self.csv_results, df_new], ignore_index=True
                )
            else:
                master = df_new.copy()
            master = master.drop_duplicates(subset="image", keep="last")
            master.to_csv(
                self.master_path,
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
                index=False,
            )
            self.csv_results = master
        self._status("Inference complete!")
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        self.view_heatmaps_btn.setEnabled(True)
        self.view_3d_btn.setEnabled(True)

    def _handle_error(self, error):
        msg = str(error)
        if "OOM" in msg or isinstance(error, tf.errors.ResourceExhaustedError):
            status(
                "Inference error: GPU memory exhausted. Try smaller batch.",
                msecs=10000,
                viewer=self.viewer,
            )
        else:
            status(
                f"Inference error: {error}", msecs=10000, viewer=self.viewer
            )
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)

    def _infer_active_layer(self):
        # ——— vérifications de base ————————————————————————————
        needed = [
            k for k in ("output", "map", "model") if self.paths.get(k) is None
        ]
        if needed:
            return self._status(f"Select folder(s): {', '.join(needed)}")

        # calques à traiter : sélection ou calque actif
        layers = list(self.viewer.layers.selection) or [
            self.viewer.layers.selection.active
        ]
        layers = [ly for ly in layers if ly is not None]
        if not layers:
            return self._status("No layer selected.")

        # ——— paramètres d’inférence ————————————————
        patch = (256, 256)
        ovlap = (128, 128)
        batch = self.batch_spin.value()
        to8 = True

        # ——— worker dans un thread pour garder l’UI réactive ————
        @thread_worker
        def worker():
            mdl = load_model(self.paths["model"])
            rows = []
            for idx, layer in enumerate(layers, start=1):
                best = infer_array(
                    data=layer.data,
                    model=mdl,
                    patch_size=patch,
                    overlap=ovlap,
                    batch_size=batch,
                    to_8bit=to8,
                )
                props, gclass = compute_statistics(best)

                # fichier heat‑map RGBA
                heat = build_heatmap_rgba(best, alpha=0.4)
                tiff.imwrite(self.paths["map"] / f"{layer.name}_map.tif", heat)

                rows.append(
                    {
                        "image": f"{layer.name}.napari",
                        "pct_connected": props[1],
                        "pct_fragmented": props[2],
                        "pct_intermediate": props[3],
                        "global_class": gclass,
                    }
                )
                yield idx  # progression

            return pd.DataFrame(rows)

        # ——— connections UI ————————————————————————————————
        total = len(layers)
        self.progress.setRange(0, total)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.layer_infer_btn.setEnabled(False)

        worker_obj = worker()
        worker_obj.yielded.connect(self.progress.setValue)
        worker_obj.returned.connect(self._handle_finished)
        worker_obj.errored.connect(self._handle_error)
        worker_obj.finished.connect(
            lambda: (
                self.progress.setVisible(False),
                self.layer_infer_btn.setEnabled(True),
            )
        )
        worker_obj.start()

    from pathlib import Path

    from tifffile import imread

    def _show_heatmaps(self):
        """Display only the heat‑maps generated in the last inference session,
        and ensure their raw images are loaded/stacked if available."""
        # 1) Pre‑checks
        if self.paths.get("map") is None:
            return self._status("Please select the ‘Map dir’ first.")
        if self.df_new is None:
            return self._status("Run an inference first, then show heat‑maps.")

        input_dir = self.paths.get(
            "input"
        )  # might be None for layer‑only runs

        # 2) Which stems were just processed?
        stems = {Path(fname).stem for fname in self.df_new["image"]}

        # 3) Remove any previously displayed map layers
        for lyr in list(self.viewer.layers):
            if lyr.name.endswith("_map"):
                self.viewer.layers.remove(lyr)

        loaded = 0
        for stem in sorted(stems):
            map_path = self.paths["map"] / f"{stem}_map.tif"
            if not map_path.exists():
                continue  # skip missing or stale files

            # 4) Add the RGBA heat‑map
            heat = imread(map_path)  # shape (H, W, 4)
            heat_layer = self.viewer.add_image(
                heat,
                name=f"{stem}_map",
                rgb=True,
                blending="translucent",
            )

            # 5) If raw layer exists, stack above it; otherwise try loading it
            if stem in self.viewer.layers:
                i_raw = self.viewer.layers.index(self.viewer.layers[stem])
                i_heat = self.viewer.layers.index(heat_layer)
                self.viewer.layers.move(i_heat, i_raw + 1)
            elif input_dir is not None:
                # find a matching raw file next to the maps
                candidates = [
                    p
                    for p in Path(input_dir).glob(f"{stem}.*")
                    if p.suffix.lower() in {".tif", ".tiff", ".stk", ".png"}
                ]
                if candidates:
                    raw_data = imread(candidates[0])
                    raw_layer = self.viewer.add_image(
                        raw_data,
                        name=stem,
                        colormap="gray",
                        blending="translucent",
                        opacity=1.0,
                    )
                    # then stack the map above it
                    i_raw = self.viewer.layers.index(raw_layer)
                    i_heat = self.viewer.layers.index(heat_layer)
                    self.viewer.layers.move(i_heat, i_raw + 1)

            loaded += 1

        # 6) Final status message
        if loaded:
            self._status(f"{loaded} new heat‑map layer(s) loaded.")
        else:
            self._status("No new heat‑maps found to display.")

    def _show_3d(self):
        if self.df_new is None and self.csv_results is None:
            return self._status("No data to plot.")
        import plotly.express as px

        source = self.graph_source_combo.currentText()
        df_plot = (
            self.csv_results.copy()
            if source == "Master Data" and self.csv_results is not None
            else self.df_new.copy()
        )
        label_map = {1: "Connected", 2: "Fragmented", 3: "Intermediate"}
        df_plot["Class"] = (
            df_plot["global_class"].map(label_map).fillna("background")
        )
        fig = px.scatter_3d(
            df_plot,
            x="pct_connected",
            y="pct_fragmented",
            z="pct_intermediate",
            hover_name="image",
            color="Class",
            color_discrete_map={
                "Connected": "red",
                "Fragmented": "green",
                "Intermediate": "blue",
                "background": "gray",
            },
            title=f"3D distribution ({source})",
        )
        out_path = self.paths["output"] / "graph3d.html"
        fig.write_html(str(out_path), include_plotlyjs="cdn")
        webbrowser.open(out_path.as_uri())
