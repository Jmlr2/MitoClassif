# src/mitoclass/_widget.py

import shutil
import webbrowser
from pathlib import Path

import pandas as pd
import tensorflow as tf
from napari.qt.threading import thread_worker
from qtpy.QtCore import QStandardPaths
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedLayout,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from tifffile import imread

from ._utils import status


class MitoclassWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self._status = lambda msg, msecs=10000: status(msg, msecs, self.viewer)
        # ———Internal states ———
        self.original_stacks = []  # for annotation
        self.viewer = napari_viewer
        self.csv_results = None  # master DataFrame
        self.df_new = None  # current session results

        # ———Technical file for the master CSV ———
        appdata = QStandardPaths.writableLocation(
            QStandardPaths.AppDataLocation
        )
        self.master_dir = Path(appdata) / "mitoclass"
        self.master_dir.mkdir(parents=True, exist_ok=True)
        self.master_path = self.master_dir / "predictions.csv"

        # Loading the master CSV if it exists
        if self.master_path.exists():
            try:
                self.csv_results = pd.read_csv(
                    self.master_path,
                    sep=";",
                    decimal=",",
                    encoding="utf-8-sig",
                )
            except Exception as e:  # noqa: BLE001
                # Error message in the status bar
                self._status(f"Error reading predictions.csv: {e}")
                self.csv_results = None

        # ——— Paths for inference ———
        self.paths = {
            "input": None,
            "output": None,
            "map": None,
            "model": None,
        }

        # ——— Training Paths ———
        # pp_input/pp_output for the pre-processing part,
        # tr_patches/tr_output for the training part,
        # pretrained_model for optional fine-tuning.

        self.paths_train = {
            "pp_input": None,
            "pp_output": None,
            "tr_patches": None,
            "tr_output": None,
            "pretrained_model": None,
        }

        # ——— Paths & state for annotation———
        self.paths_annot = {
            "raw_input": None,
            "annot_output": None,
        }
        self.annotations = {}  # {filename: class_label}
        self.annot_index = 0  # current index for annotation

        # ———“Model name” field for training ———
        self.model_name_le = QLineEdit("new_model")

        # Building the interface
        self._build_ui()
        default_model = Path(__file__).parent / "models" / "base_model.h5"
        if default_model.exists():
            self.paths["model"] = default_model
            self.model_path_btn.setText(f"Model: {default_model.name}")

    def _build_ui(self):
        # ——— Top buttons ———
        self.btn_annotate = QPushButton("Annotate")
        self.btn_train = QPushButton("Train model")
        self.btn_infer = QPushButton("Prediction")

        self.add_to_master_cb = QCheckBox("Add to master")
        self.clear_master_btn = QPushButton("Clear master")

        h_top = QHBoxLayout()
        h_top.addWidget(self.btn_annotate)
        h_top.addWidget(self.btn_train)
        h_top.addWidget(self.btn_infer)

        # ——— Training tabs ———
        self.train_tabs = QTabWidget()

        # — Preprocessing tab —
        self.tab_pp = QWidget()
        pp_layout = QFormLayout(self.tab_pp)

        # 1) Folder selection
        self.pp_input_btn = QPushButton("Raw data folder")
        self.pp_output_btn = QPushButton("Patches output folder")

        # 2) Preprocessing parameters (names suffixed _pp)
        self.bits_pp_combo = QComboBox()
        self.bits_pp_combo.addItems(["8-bit", "16-bit"])
        self.split_pp_spin = QSpinBox()
        self.split_pp_spin.setRange(1, 99)
        self.split_pp_spin.setValue(70)
        self.split_pp_spin_val = QSpinBox()
        self.split_pp_spin_val.setRange(1, 99)
        self.split_pp_spin_val.setValue(15)
        self.patch_pp_spin = QSpinBox()
        self.patch_pp_spin.setRange(1, 4096)
        self.patch_pp_spin.setValue(128)
        self.ov_pp_spin = QSpinBox()
        self.ov_pp_spin.setRange(0, 4096)
        self.ov_pp_spin.setValue(64)
        self.minpix_pp_spin = QSpinBox()
        self.minpix_pp_spin.setRange(0, 100000)
        self.minpix_pp_spin.setValue(100)

        # 3) Button and progress bar
        self.pp_run_btn = QPushButton("Run preprocessing")
        self.pp_prog = QProgressBar()
        self.pp_prog.setVisible(False)

        # --- Form layout assembly ---
        pp_layout.addRow("Raw dir:", self.pp_input_btn)
        pp_layout.addRow("Output dir:", self.pp_output_btn)
        pp_layout.addRow("Bit depth:", self.bits_pp_combo)
        pp_layout.addRow("Split train %:", self.split_pp_spin)
        pp_layout.addRow("Split val %:", self.split_pp_spin_val)
        pp_layout.addRow("Patch size:", self.patch_pp_spin)
        pp_layout.addRow("Overlap:", self.ov_pp_spin)
        pp_layout.addRow("Min mask pixels:", self.minpix_pp_spin)
        pp_layout.addRow(self.pp_run_btn)
        pp_layout.addRow(self.pp_prog)

        self.train_tabs.addTab(self.tab_pp, "Preprocessing")

        # — Tab Training —
        self.tab_tr = QWidget()
        tr_layout = QFormLayout(self.tab_tr)

        # 1) Selection of files and models
        self.tr_patches_btn = QPushButton("Patches root folder")
        self.tr_output_btn = QPushButton("Model output folder")
        self.pretrained_btn = QPushButton("Choose initial model (.h5)")
        self.clear_pretrained_btn = QPushButton("Clear")

        # 2) Training parameters (names suffixed with _tr)
        self.model_name_le = QLineEdit("new_model")
        tr_layout.addRow("Model name:", self.model_name_le)
        self.bits_tr_combo = QComboBox()
        self.bits_tr_combo.addItems(["8-bit", "16-bit"])
        self.patch_tr_spin = QSpinBox()
        self.patch_tr_spin.setRange(1, 4096)
        self.patch_tr_spin.setValue(128)
        self.ov_tr_spin = QSpinBox()
        self.ov_tr_spin.setRange(0, 4096)
        self.ov_tr_spin.setValue(64)
        self.batch_tr_spin = QSpinBox()
        self.batch_tr_spin.setRange(1, 1024)
        self.batch_tr_spin.setValue(32)
        self.epoch_tr_spin = QSpinBox()
        self.epoch_tr_spin.setRange(1, 1000)
        self.epoch_tr_spin.setValue(100)
        self.lr_tr_spin = QDoubleSpinBox()
        self.lr_tr_spin.setRange(1e-6, 1.0)
        self.lr_tr_spin.setDecimals(6)
        self.lr_tr_spin.setValue(5e-4)
        self.patience_tr_spin = QSpinBox()
        self.patience_tr_spin.setRange(1, 100)
        self.patience_tr_spin.setValue(10)
        self.cb_delete_patches = QCheckBox("Delete patches after training")
        self.train_btn_exec = QPushButton("Start training")
        self.train_prog = QProgressBar()
        self.train_prog.setVisible(False)

        # 3) Container for pretrained + clear
        container_pretrained = QWidget()
        hpre = QHBoxLayout(container_pretrained)
        hpre.setContentsMargins(0, 0, 0, 0)
        hpre.addWidget(self.pretrained_btn)
        hpre.addWidget(self.clear_pretrained_btn)

        # --- Form layout assembly ---
        tr_layout.addRow("Patches dir:", self.tr_patches_btn)
        tr_layout.addRow("Model out dir:", self.tr_output_btn)
        tr_layout.addRow("Initial model (opt.):", container_pretrained)
        tr_layout.addRow("Bit depth:", self.bits_tr_combo)
        tr_layout.addRow("Patch size:", self.patch_tr_spin)
        tr_layout.addRow("Overlap:", self.ov_tr_spin)
        tr_layout.addRow("Batch size:", self.batch_tr_spin)
        tr_layout.addRow("Epochs:", self.epoch_tr_spin)
        tr_layout.addRow("Learning rate:", self.lr_tr_spin)
        tr_layout.addRow("Patience:", self.patience_tr_spin)
        tr_layout.addRow(self.cb_delete_patches)
        tr_layout.addRow(self.train_btn_exec)
        tr_layout.addRow(self.train_prog)

        self.train_tabs.addTab(self.tab_tr, "Training")

        # ——— Inference page ———
        self.page_infer = QWidget()
        form_inf = QFormLayout(self.page_infer)
        self.input_dir_btn = QPushButton("Choose input folder")
        self.output_dir_btn = QPushButton("Choose output folder")
        self.map_dir_btn = QPushButton("Select heatmaps folder")
        self.model_path_btn = QPushButton("Choose model .h5")
        self.patch_inf_spin = QSpinBox()
        self.patch_inf_spin.setRange(1, 4096)
        self.patch_inf_spin.setValue(256)
        self.ov_inf_spin = QSpinBox()
        self.ov_inf_spin.setRange(0, 4096)
        self.ov_inf_spin.setValue(128)
        self.bits_inf_combo = QComboBox()
        self.bits_inf_combo.addItems(["8‑bit", "16‑bit"])
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setValue(16)
        self.run_btn = QPushButton("Run Inference")
        self.graph_source_combo = QComboBox()
        self.graph_source_combo.addItems(["Master Data", "Last Session"])
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.view_heatmaps_btn = QPushButton("Show heatmaps")
        self.view_heatmaps_btn.setEnabled(False)
        self.view_3d_btn = QPushButton("Display 3D graph")
        self.view_3d_btn.setEnabled(False)
        self.layer_infer_btn = QPushButton("Active layer inference (Napari)")

        # Assemble inference form
        form_inf.addRow("Input dir:", self.input_dir_btn)
        form_inf.addRow("Output dir:", self.output_dir_btn)
        form_inf.addRow("Map dir:", self.map_dir_btn)
        form_inf.addRow("Model file:", self.model_path_btn)
        form_inf.addRow("Patch size:", self.patch_inf_spin)
        form_inf.addRow("Overlap:", self.ov_inf_spin)
        form_inf.addRow("Bit depth:", self.bits_inf_combo)
        form_inf.addRow("Batch size:", self.batch_spin)
        form_inf.addRow("3D graph source:", self.graph_source_combo)
        # New controls for master handling
        form_inf.addRow(self.add_to_master_cb)
        form_inf.addRow(self.clear_master_btn)
        form_inf.addRow(self.run_btn)
        form_inf.addRow(self.progress)
        form_inf.addRow(self.view_heatmaps_btn, self.view_3d_btn)
        form_inf.addRow(self.layer_infer_btn)

        # ——— Annotation page ———
        self.page_annotate = QWidget()
        form_annot = QFormLayout(self.page_annotate)
        self.raw_input_btn = QPushButton("Choose input folder")
        self.annot_output_btn = QPushButton("Select output folder annotations")
        self.classes_le = QLineEdit()
        self.buttons_widget = QWidget()
        self.buttons_layout = QHBoxLayout(self.buttons_widget)
        self.annot_status_lbl = QLabel("0 / 0 annotated")
        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Following")
        self.export_btn = QPushButton("Export classifications")
        self.annot_prog = QProgressBar()
        self.annot_prog.setVisible(False)
        self.clear_annotations_btn = QPushButton("Clear annotations")
        self.move_files_cb = QCheckBox("Move files instead of copy")

        form_annot.addRow("Input folder:", self.raw_input_btn)
        form_annot.addRow("Annotations output folder:", self.annot_output_btn)
        form_annot.addRow("Classes (separated by ‘;’):", self.classes_le)
        form_annot.addRow("", self.buttons_widget)
        form_annot.addRow(self.annot_status_lbl)
        form_annot.addRow(self.prev_btn, self.next_btn)
        form_annot.addRow(self.move_files_cb)
        form_annot.addRow(self.export_btn)
        form_annot.addRow(self.annot_prog)
        form_annot.addRow(self.clear_annotations_btn)

        # ——— Stacked layout ———
        self.stack = QStackedLayout()
        self.stack.addWidget(
            self.train_tabs
        )  # index 0 = training/preprocessing
        self.stack.addWidget(self.page_infer)  # index 1 = inference
        self.stack.addWidget(self.page_annotate)  # index 2 = annotation

        # ——— Main layout ———
        main = QVBoxLayout(self)
        main.addLayout(h_top)
        main.addLayout(self.stack)

        # ——— Connections ———
        # Top navigation
        self.btn_train.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        self.btn_infer.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        self.btn_annotate.clicked.connect(
            lambda: self.stack.setCurrentIndex(2)
        )
        # Preprocessing tab
        self.pp_input_btn.clicked.connect(
            lambda: self._choose_dir_train("pp_input")
        )
        self.pp_output_btn.clicked.connect(
            lambda: self._choose_dir_train("pp_output")
        )
        self.pp_run_btn.clicked.connect(self._run_preprocessing)
        # Training tab
        self.tr_patches_btn.clicked.connect(
            lambda: self._choose_dir_train("tr_patches")
        )
        self.tr_output_btn.clicked.connect(
            lambda: self._choose_dir_train("tr_output")
        )
        self.pretrained_btn.clicked.connect(
            lambda: self._choose_file_train("pretrained_model")
        )
        self.clear_pretrained_btn.clicked.connect(self._clear_pretrained_model)
        self.train_btn_exec.clicked.connect(self._run_training_only)
        # Inference tab
        self.btn_infer.clicked.connect(
            lambda: self.stack.setCurrentWidget(self.page_infer)
        )
        self.input_dir_btn.clicked.connect(lambda: self._choose_dir("input"))
        self.output_dir_btn.clicked.connect(lambda: self._choose_dir("output"))
        self.map_dir_btn.clicked.connect(lambda: self._choose_dir("map"))
        self.model_path_btn.clicked.connect(self._choose_model)
        self.run_btn.clicked.connect(self._run_inference)
        self.view_heatmaps_btn.clicked.connect(self._show_heatmaps)
        self.view_3d_btn.clicked.connect(self._show_3d)
        self.clear_master_btn.clicked.connect(self._clear_master)
        self.layer_infer_btn.clicked.connect(self._infer_active_layer)
        # Annotation tab
        self.raw_input_btn.clicked.connect(
            lambda: self._choose_dir_annot("raw_input")
        )
        self.annot_output_btn.clicked.connect(
            lambda: self._choose_dir_annot("annot_output")
        )
        self.classes_le.editingFinished.connect(self._update_classes)
        self.prev_btn.clicked.connect(lambda: self._step_annotation(-1))
        self.next_btn.clicked.connect(lambda: self._step_annotation(1))
        self.export_btn.clicked.connect(self._export_annotations)
        self.clear_annotations_btn.clicked.connect(self._clear_annotations)

    def _infer_active_layer(self):
        """Run Mitoclass on the active layer in Napari."""
        from ._actions import infer_selected_layer

        # UI parameters
        patch_size = (self.patch_inf_spin.value(),) * 2
        overlap = (self.ov_inf_spin.value(),) * 2
        batch_sz = self.batch_spin.value()
        to_8bit = self.bits_inf_combo.currentText().startswith("8")

        model_path = self.paths.get("model")
        if model_path is None:
            self._status(
                "Please choose a model (.h5) in the Inference tab first."
            )
            return

        stats = infer_selected_layer(
            viewer=self.viewer,
            model_path=model_path,
            patch_size=patch_size,
            overlap=overlap,
            batch_size=batch_sz,
            to_8bit=to_8bit,
            add_table=True,
        )
        if stats is None:
            self._status("Mitoclass inference failed (see console).")
            return

        # quick status message with dominant class
        cls = stats["global_class"]
        pct = stats["proportions"][cls]
        self._status(f"Mitoclass: class {cls} ({pct:.1f}%) on active layer.")

    def _choose_dir_train(self, key):
        """
        Generic directory picker for both preprocessing and training tabs.
        key must be one of: 'pp_input', 'pp_output', 'tr_patches', 'tr_output'
        """
        d = QFileDialog.getExistingDirectory(self, "Select a folder")
        if not d:
            return

        # store path
        self.paths_train[key] = Path(d)

        # update corresponding button text
        btn = getattr(self, f"{key}_btn", None)
        if btn is not None:
            label = key.replace("_", " ").title()
            btn.setText(f"{label}: {d}")

    def _choose_file_train(self, key, file_filter="*.h5"):
        """
        File picker for the optional pretrained model.
        key must be 'pretrained_model'.
        """
        fpath, _ = QFileDialog.getOpenFileName(
            self, "Select a model (.h5)", filter=file_filter
        )
        if not fpath:
            return

        self.paths_train[key] = Path(fpath)
        # update the button in the Training tab
        self.pretrained_btn.setText(f"Initial model: {Path(fpath).name}")
        # optional quick status
        self._status(f"Loaded initial model: {Path(fpath).name}")

    def _clear_pretrained_model(self):
        """
        Clear the pretrained_model setting and reset button text.
        """
        self.paths_train["pretrained_model"] = None
        self.pretrained_btn.setText("Choose initial model (.h5)")
        self._status("Training from scratch (no initial model).")

    def _run_preprocessing(self):
        pp_in = self.paths_train.get("pp_input")
        pp_out = self.paths_train.get("pp_output")
        if pp_in is None or pp_out is None:
            return self._status(
                "Please select the raw and output folders for the patches."
            )

        @thread_worker(
            connect={
                "started": lambda: (
                    self.pp_prog.setRange(0, 0),
                    self.pp_prog.setVisible(True),
                ),
                "returned": lambda _: self._handle_pp_finished(),
                "errored": lambda e: self._handle_pp_error(e),
            }
        )
        def worker():
            from ._pretreat import preprocess

            # rebuild the splits
            p_train = self.split_pp_spin.value() / 100
            p_val = self.split_pp_spin_val.value() / 100
            p_test = 1.0 - p_train - p_val

            # cutting parameters
            patch_size = (self.patch_pp_spin.value(),) * 2
            overlap = (self.ov_pp_spin.value(),) * 2
            min_pix = self.minpix_pp_spin.value()
            to_8bit = self.bits_pp_combo.currentText().startswith("8")

            preprocess(
                input_dir=pp_in,
                output_dir=pp_out,
                splits=(p_train, p_val, p_test),
                patch_size=patch_size,
                overlap=overlap,
                min_mask_pixels=min_pix,
                to_8bit=to_8bit,
                seed=42,
            )
            return True

        worker()

    def _handle_pp_finished(self):
        """Called when preprocessing completes successfully."""
        self.pp_prog.setVisible(False)
        # make patches folder available for the Training tab
        self.paths_train["tr_patches"] = self.paths_train["pp_output"]
        self.tr_patches_btn.setText(
            f"Patches: {self.paths_train['tr_patches']}"
        )
        self._status("Preprocessing complete.")

    def _handle_pp_error(self, error):
        """Called if preprocessing fails."""
        self.pp_prog.setVisible(False)
        self._status(f"Preprocessing error: {error}")

    def _run_training_only(self):
        patches = self.paths_train.get("tr_patches")
        outdir = self.paths_train.get("tr_output")
        if patches is None or outdir is None:
            return self._status(
                "Please select patches folder and model output folder first."
            )

        def on_train_started():
            self.train_prog.setRange(0, 0)
            self.train_prog.setVisible(True)
            self.train_btn_exec.setEnabled(False)

        def on_train_finished(res):
            self.train_prog.setVisible(False)
            self.train_btn_exec.setEnabled(True)
            self._handle_train_finished(res)

        def on_train_error(e):
            self.train_prog.setVisible(False)
            self.train_btn_exec.setEnabled(True)
            self._handle_train_error(e)

        @thread_worker(
            connect={
                "started": on_train_started,
                "returned": on_train_finished,
                "errored": on_train_error,
            }
        )
        def worker():
            from ._trainer import train_pipeline_from_patches

            to_8bit = self.bits_tr_combo.currentText().startswith("8")
            patch_size = (self.patch_tr_spin.value(),) * 2
            batch_sz = self.batch_tr_spin.value()
            epochs = self.epoch_tr_spin.value()
            lr = self.lr_tr_spin.value()
            patience = self.patience_tr_spin.value()
            model_name = self.model_name_le.text()
            delete = self.cb_delete_patches.isChecked()

            return train_pipeline_from_patches(
                patches_root=patches,
                output_dir=outdir,
                to_8bit=to_8bit,
                patch_size=patch_size,
                batch_size=batch_sz,
                epochs=epochs,
                learning_rate=lr,
                patience=patience,
                model_name=model_name,
                pretrained_model=self.paths_train.get("pretrained_model"),
                delete_patches=delete,
            )

        self.train_worker = worker()

    def _handle_train_finished(self, res):
        _, model_path, (test_loss, test_acc) = res
        self._status(
            f"Training finished — Model: {model_path.name} | "
            f"Loss: {test_loss:.4f}, Acc: {test_acc:.4f}"
        )
        self.train_prog.setVisible(False)
        self.train_btn_exec.setEnabled(True)

    def _handle_train_error(self, error):
        """
        Handle training errors; if GPU OOM, suggest restart or smaller batch.
        """
        msg = str(error)
        if "OOM" in msg or isinstance(error, tf.errors.ResourceExhaustedError):
            status(
                "Training error: GPU memory exhausted. "
                "You may want to restart Napari or try a smaller batch size.",
                msecs=10000,
                viewer=self.viewer,
            )
        else:
            status(f"Training error: {error}", msecs=10000, viewer=self.viewer)

        self.train_prog.setVisible(False)
        self.train_btn_exec.setEnabled(True)

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
        # 1) Path verification (input, output, map and model are all required)
        missing = [
            k
            for k in ("input", "output", "map", "model")
            if self.paths.get(k) is None
        ]
        if missing:
            self._status(f"Missing path(s): {', '.join(missing)}")
            return

        from ._processor import process_folder

        # 2) UI Settings
        images = [
            p
            for p in sorted(self.paths["input"].iterdir())
            if p.suffix.lower() in {".tif", ".tiff", ".stk", ".png"}
        ]
        total = len(images)
        if total == 0:
            self._status("No images found in input folder.")
            return

        patch_size = (self.patch_inf_spin.value(),) * 2
        overlap = (self.ov_inf_spin.value(),) * 2
        batch_size = self.batch_spin.value()
        to_8bit = self.bits_inf_combo.currentText().startswith("8")

        # 3) Configuring the progress bar
        self.progress.setRange(0, total)
        self.progress.setValue(0)
        self.progress.setVisible(True)
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
        # Always save new session
        if self.add_to_master_cb.isChecked():
            # merge into master
            if self.master_path.exists():
                master_df = pd.read_csv(
                    self.master_path,
                    sep=";",
                    decimal=",",
                    encoding="utf-8-sig",
                )
                master_df = pd.concat([master_df, df_new], ignore_index=True)
                master_df = master_df.drop_duplicates(
                    subset="image", keep="last"
                )
            else:
                master_df = df_new.copy()
            master_df.to_csv(
                self.master_path,
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
                index=False,
            )
            self.csv_results = master_df
        self._status("Inference complete!")
        self.view_heatmaps_btn.setEnabled(True)
        self.view_3d_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)

    def _handle_error(self, error):
        """Handle inference errors; if GPU OOM, suggest restart or smaller batch."""
        msg = str(error)
        if "OOM" in msg or isinstance(error, tf.errors.ResourceExhaustedError):
            status(
                "Inference error: GPU memory exhausted. "
                "You may want to restart Napari or try a smaller batch size.",
                msecs=10000,
                viewer=self.viewer,
            )
        else:
            status(
                f"Inference error: {error}", msecs=10000, viewer=self.viewer
            )

        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)

    def _clear_master(self):
        try:
            if self.master_path.exists():
                self.master_path.unlink()
            self.csv_results = None
            self._status("Master CSV cleared.")
        except Exception as e:  # noqa: BLE001
            self._status(f"Error clearing master: {e}")

    def _show_heatmaps(self):
        for tif in sorted(self.paths["map"].glob("*_map.tif")):
            img = imread(tif)
            layer = self.viewer.add_image(img, name=tif.stem, rgb=True)
            # layer.colormap = "gray"

    def _show_3d(self):
        if self.csv_results is None or self.df_new is None:
            return self._status("No results: run inference first.")

        import plotly.express as px

        source = self.graph_source_combo.currentText()
        df_plot = (
            self.csv_results if source == "Master Data" else self.df_new
        ).copy()

        # Mapping texte → couleur
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
            title=f"3D distribution of classes ({source})",
        )

        # 3) Saving and displaying the graph
        self.master_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.paths["output"] / "graph3d.html"
        fig.write_html(str(out_path), include_plotlyjs="cdn")
        webbrowser.open(out_path.as_uri())

    def _choose_dir_annot(self, key):
        folder = QFileDialog.getExistingDirectory(self, "Select a folder")
        if not folder:
            return

        self.paths_annot[key] = Path(folder)

        if key == "raw_input":
            # update button text
            self.raw_input_btn.setText(f"Raw folder: {folder}")

            # load files and reset annotations
            files = sorted(
                p
                for p in Path(folder).iterdir()
                if p.suffix.lower() in {".tif", ".tiff", ".stk", ".png"}
            )
            self.original_stacks = files
            self.annotations.clear()
            self.annot_index = 0
            self.annot_status_lbl.setText(f"0 / {len(files)} annotated")
            self._update_classes()
            self._display_current()

        elif key == "annot_output":
            self.annot_output_btn.setText(f"Annotations folder: {folder}")
            # optionally you could load an existing JSON here:
            # from ._annotator import load_annotation
            # annos = load_annotation(Path(folder) / "annotations.json")
            # if annos:
            #     self.annotations = annos
            #     self.annot_status_lbl.setText(f"{len(annos)} / {len(self.original_stacks)} annotated")
            #     self._display_current()

    def _update_classes(self):
        """(Re)built the class button area."""
        # 1) Retrieve the labels
        classes = [
            c.strip() for c in self.classes_le.text().split(";") if c.strip()
        ]

        # 2) Empty the old buttons
        for i in reversed(range(self.buttons_layout.count())):
            w = self.buttons_layout.itemAt(i).widget()
            if w:
                w.setParent(None)

        # 3) Create one button per class
        for cls in classes:
            btn = QPushButton(cls)
            btn.clicked.connect(lambda _, c=cls: self._assign_and_next(c))
            self.buttons_layout.addWidget(btn)

    def _assign_and_next(self, cls_label: str):
        if not self.original_stacks:
            return

        img = self.original_stacks[self.annot_index]
        first_time = img.name not in self.annotations
        self.annotations[img.name] = cls_label

        if first_time:
            done, total = len(self.annotations), len(self.original_stacks)
            self.annot_status_lbl.setText(f"{done} / {total} annotated")

        # Advance then display the new image
        self.annot_index += 1
        self._display_current()

    def _export_annotations(self):
        from pathlib import Path

        from napari.qt.threading import thread_worker

        # 1) Check that input and output dirs are set
        raw_dir = self.paths_annot.get("raw_input")
        out_dir = self.paths_annot.get("annot_output")

        if raw_dir is None:
            self._status("Please select a raw input folder first.")
            return

        if out_dir is None:
            self._status("Please select an annotation output folder first.")
            return

        # 2) Check that there is something to export
        if not self.annotations:
            self._status("No annotations to export.")
            return

        total = len(self.annotations)

        # 3) Run export in background thread
        @thread_worker(
            connect={
                "started": lambda: self.annot_prog.setVisible(True),
                "yielded": lambda i: self.annot_prog.setValue(
                    int((i / total) * 100)
                ),
                "returned": lambda _: [
                    self.annot_prog.setVisible(False),
                    self._status(
                        "Export complete! "
                        + (
                            "Files moved."
                            if self.move_files_cb.isChecked()
                            else "Files copied."
                        )
                    ),
                ],
                "errored": lambda e: [
                    self.annot_prog.setVisible(False),
                    self._status(f"Export error: {e}"),
                ],
            }
        )
        def worker():
            i = 0
            for fname, cls in self.annotations.items():
                try:
                    src = Path(raw_dir) / fname
                    dst = Path(out_dir) / cls / fname
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if self.move_files_cb.isChecked():
                        shutil.move(src, dst)
                    else:
                        shutil.copy2(src, dst)

                except Exception as e:  # noqa: BLE001
                    print(f"Error copying '{fname}': {e}")
                finally:
                    i += 1
                    yield i
            return True

        worker()

    def _display_current(self):
        """Displays the image at self.annot_index and highlights its annotation."""
        if not self.original_stacks:
            return

        # 1) Clamp and wrap around
        n = len(self.original_stacks)
        self.annot_index %= n
        img_path = self.original_stacks[self.annot_index]

        # 2) Display in Napari
        stack = imread(img_path)
        self.viewer.layers.clear()
        self.viewer.add_image(stack, name=img_path.name)

        # 3) Update status
        done = len(self.annotations)
        self.annot_status_lbl.setText(f"{done} / {n} annotated")

        # 4) Rebuilding the buttons and highlighting
        # (we use the same method as _update_classes to create the buttons)

        self._update_classes()
        current_label = self.annotations.get(img_path.name)
        for btn in self.buttons_widget.findChildren(QPushButton):
            # if it is the button of the current class → we underline it
            if btn.text() == current_label:
                btn.setStyleSheet(
                    "font-weight: bold; text-decoration: underline;"
                )
            else:
                btn.setStyleSheet("")

    def _step_annotation(self, delta: int):
        """Moves the index by +1 or -1 and redisplays."""
        if not self.original_stacks:
            return
        self.annot_index += delta
        self._display_current()

    def _clear_annotations(self):
        """Clear all annotations and restart from the first image."""
        if not self.original_stacks:
            return
        self.annotations.clear()
        self.annot_index = 0
        self.annot_status_lbl.setText(
            f"0 / {len(self.original_stacks)} annotated"
        )
        self._display_current()
        self._status("Annotations cleared. Starting over.")
