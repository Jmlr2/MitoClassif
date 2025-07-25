# src/mitoclass/_trainer.py

import random
import shutil
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from ._pretreat import preprocess
from ._utils import status


def train_pipeline_from_patches(
    patches_root: Union[str, Path],
    output_dir: Union[str, Path],
    patch_size: Tuple[int, int],
    batch_size: int = 32,
    epochs: int = 100,
    learning_rate: float = 5e-4,
    patience: int = 10,
    model_name: str = "best_model",
    pretrained_model: Union[str, Path, None] = None,
    delete_patches: bool = False,
    to_8bit: bool = False,
) -> Tuple[dict, Path, Tuple[float, float]]:
    """
    Trains a model from an existing patch directory
    (structure: train/, val/, test/ + manifest.csv).
    """
    patches_root = Path(patches_root)
    train_dir = patches_root / "train"
    val_dir = patches_root / "val"
    test_dir = patches_root / "test"

    # --- Generators with normalization and (for train) data augmentation D4 ---
    def apply_d4(img: np.ndarray) -> np.ndarray:
        choice = random.randint(0, 7)
        if choice == 0:
            return img
        ops = [
            lambda x: np.rot90(x, k=1),
            lambda x: np.rot90(x, k=2),
            lambda x: np.rot90(x, k=3),
            lambda x: np.fliplr(x),
            lambda x: np.flipud(x),
            lambda x: np.rot90(np.fliplr(x), k=1),
            lambda x: np.rot90(np.flipud(x), k=1),
        ]
        return ops[choice - 1](img)

    def normalize(img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        if to_8bit:
            return img / 255.0
        return img / 65535.0

    train_gen = ImageDataGenerator(
        preprocessing_function=lambda img: normalize(apply_d4(img))
    ).flow_from_directory(
        str(train_dir),
        target_size=patch_size,
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=42,
    )

    val_gen = ImageDataGenerator(
        preprocessing_function=normalize
    ).flow_from_directory(
        str(val_dir),
        target_size=patch_size,
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    # --- Calculating class_weights ---
    num_classes = len(train_gen.class_indices)
    labels = train_gen.classes
    cw = compute_class_weight(
        "balanced", classes=np.arange(num_classes), y=labels
    )
    class_weights = dict(enumerate(cw))  # C416

    # --- Building the model ---
    sample_x, _ = next(train_gen)
    input_shape = sample_x.shape[1:]
    if pretrained_model:
        model = tf.keras.models.load_model(pretrained_model, compile=False)
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
    else:
        model = Sequential(
            [
                Conv2D(64, (3, 3), activation="relu", input_shape=input_shape),
                MaxPooling2D((2, 2)),
                Conv2D(128, (3, 3), activation="relu"),
                MaxPooling2D((2, 2)),
                Conv2D(256, (3, 3), activation="relu"),
                MaxPooling2D((2, 2)),
                Conv2D(512, (3, 3), activation="relu"),
                MaxPooling2D((2, 2)),
                Conv2D(512, (3, 3), activation="relu"),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(1024, activation="relu"),
                Dropout(0.5),
                Dense(num_classes, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    # --- Callbacks and training ---
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{model_name}.h5"
    checkpoint = callbacks.ModelCheckpoint(
        str(model_path), save_best_only=True, monitor="val_accuracy"
    )
    earlystop = callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=[checkpoint, earlystop],
    )
    pd.DataFrame(history.history).to_csv(
        output_dir / f"{model_name}_history.csv",
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        index=False,
    )

    # --- Evaluation on the test set ---
    test_gen = ImageDataGenerator(
        preprocessing_function=normalize
    ).flow_from_directory(
        str(test_dir),
        target_size=patch_size,
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)

    # --- Classification repor ---
    y_true = test_gen.classes
    test_gen.reset()
    y_proba = model.predict(test_gen, verbose=0)
    y_pred = y_proba.argmax(axis=1)
    idx_to_class = {v: k for k, v in test_gen.class_indices.items()}
    target_names = [idx_to_class[i] for i in sorted(idx_to_class)]
    classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True
    )

    # --- Save reports ---
    import json

    report_dict = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True
    )

    report_txt = json.dumps(
        report_dict, indent=4
    )  # Convert to JSON formatted string

    (output_dir / f"{model_name}_classification_report.txt").write_text(
        "Classification Report\n====================\n\n" + report_txt
    )

    pd.DataFrame(
        {
            "test_loss": [test_loss],
            "test_accuracy": [test_acc],
        }
    ).to_csv(
        output_dir / f"{model_name}_test_metrics.csv",
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        index=False,
    )

    # --- Optional deletion of train/val/test subfolders ---
    if delete_patches:
        for subset in ("train", "val", "test"):
            dir_to_remove = patches_root / subset
            if dir_to_remove.exists() and dir_to_remove.is_dir():
                try:
                    shutil.rmtree(dir_to_remove)
                except Exception as e:  # noqa: BLE001
                    status(f"Unable to delete {dir_to_remove}: {e}")
    return history.history, model_path, (test_loss, test_acc)


def train_pipeline(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    splits: Tuple[float, float, float],
    patch_size: Tuple[int, int],
    overlap: Tuple[int, int],
    min_mask_pixels: int,
    to_8bit: bool,
    seed: int,
    batch_size: int = 32,
    epochs: int = 100,
    learning_rate: float = 0.0005,
    patience: int = 10,
    model_name: str = "best_model",
    pretrained_model: Union[str, Path, None] = None,
) -> Tuple[dict, Path, Tuple[float, float]]:
    """
    Complete pipeline: preprocessing + CNN training.
    """
    # 1. Pretreatment
    preprocess(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        splits=splits,
        patch_size=patch_size,
        overlap=overlap,
        min_mask_pixels=min_mask_pixels,
        to_8bit=to_8bit,
        seed=seed,
    )
    # 2. Training from generated patches
    return train_pipeline_from_patches(
        patches_root=output_dir,
        output_dir=output_dir,
        patch_size=patch_size,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        patience=patience,
        model_name=model_name,
        pretrained_model=pretrained_model,
        delete_patches=False,
        to_8bit=to_8bit,
    )
