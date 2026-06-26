#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import json
from typing import Optional

import numpy as np
import pandas as pd
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, Dense, Dropout, GlobalAveragePooling1D, Input, MaxPooling1D
from tensorflow.keras.models import Sequential, load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


DATASET_CANDIDATES = (
    Path("ECG_window_df.csv"),
    Path("Desktop/ECG_window_df.csv"),
    Path.home() / "Desktop" / "ECG_window_df.csv",
)

TEST_PATIENTS = {"233", "234"}
BATCH_SIZE = 256
EPOCHS = 5
RANDOM_STATE = 42
MODEL_PATH = Path("ecg_1d_cnn.keras")
FEATURE_COLUMNS_PATH = Path("feature_columns.json")
PREPROCESSING_CONFIG_PATH = Path("preprocessing_config.json")


def find_dataset_path() -> Path:
    for path in DATASET_CANDIDATES:
        if path.exists():
            return path
    candidates = "\n".join(f"- {path}" for path in DATASET_CANDIDATES)
    raise FileNotFoundError(f"ECG_window_df.csv was not found. Checked:\n{candidates}")


def load_dataset() -> pd.DataFrame:
    dataset_path = find_dataset_path()
    df = pd.read_csv(dataset_path)

    unnamed_columns = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed_columns:
        df = df.drop(columns=unnamed_columns)

    if "Results" not in df.columns:
        raise ValueError("The dataset must contain a 'Results' label column.")

    print(f"Loaded dataset: {dataset_path}")
    print(f"Dataset shape: {df.shape}")
    print("Class distribution:")
    print(df["Results"].value_counts().sort_index())

    return df


def get_voltage_columns(df: pd.DataFrame) -> list[str]:
    metadata_columns = {"Window", "Patient", "Results", "Annotation"}
    voltage_columns = [
        col
        for col in df.columns
        if col not in metadata_columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not voltage_columns:
        raise ValueError("No numeric ECG voltage columns were found.")

    return voltage_columns


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "Patient" in df.columns:
        patient_ids = df["Patient"].astype(str)
        test_mask = patient_ids.isin(TEST_PATIENTS)

        if test_mask.any() and (~test_mask).any():
            train_df = df.loc[~test_mask].copy()
            test_df = df.loc[test_mask].copy()
            print(f"Patient-based split. Test patients: {sorted(TEST_PATIENTS)}")
            return train_df, test_df

    stratify = df["Results"] if df["Results"].nunique() > 1 else None
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    print("Random stratified split. Patient-based split was not available.")
    return train_df.copy(), test_df.copy()


def prepare_arrays(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    voltage_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = train_df[voltage_columns].to_numpy(dtype=np.float32)
    x_test = test_df[voltage_columns].to_numpy(dtype=np.float32)
    y_train = train_df["Results"].to_numpy(dtype=np.float32)
    y_test = test_df["Results"].to_numpy(dtype=np.float32)

    x_train = min_max_scale_windows(x_train)
    x_test = min_max_scale_windows(x_test)

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return x_train, x_test, y_train, y_test


def min_max_scale_windows(windows: np.ndarray) -> np.ndarray:
    row_min = windows.min(axis=1, keepdims=True)
    row_max = windows.max(axis=1, keepdims=True)
    row_range = row_max - row_min
    safe_range = np.where(row_range == 0, 1.0, row_range)
    return (windows - row_min) / safe_range


def compute_training_class_weights(y_train: np.ndarray) -> Optional[dict[int, float]]:
    classes = np.unique(y_train.astype(int))
    if len(classes) < 2:
        return None

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train.astype(int),
    )
    return {int(class_id): float(weight) for class_id, weight in zip(classes, weights)}


def build_model(input_shape: tuple[int, int]) -> Sequential:
    model = Sequential(
        [
            Input(shape=input_shape),
            Conv1D(32, kernel_size=7, activation="relu"),
            Conv1D(32, kernel_size=7, activation="relu"),
            MaxPooling1D(pool_size=3),
            Dropout(0.2),
            Conv1D(64, kernel_size=5, activation="relu"),
            Conv1D(64, kernel_size=5, activation="relu"),
            MaxPooling1D(pool_size=3),
            Dropout(0.2),
            Conv1D(96, kernel_size=3, activation="relu"),
            GlobalAveragePooling1D(),
            Dense(48, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
            metrics.AUC(name="auc"),
        ],
    )
    return model


def main() -> None:
    df = load_dataset()
    voltage_columns = get_voltage_columns(df)
    train_df, test_df = split_dataset(df)
    x_train, x_test, y_train, y_test = prepare_arrays(train_df, test_df, voltage_columns)

    print(f"Voltage columns: {len(voltage_columns)}")
    print(f"Train shapes: X={x_train.shape}, y={y_train.shape}")
    print(f"Test shapes: X={x_test.shape}, y={y_test.shape}")

    model = build_model(input_shape=x_train.shape[1:])
    model.summary()
    class_weight = compute_training_class_weights(y_train)
    if class_weight:
        print(f"Class weights: {class_weight}")

    early_stopping = EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=2,
        restore_best_weights=True,
    )

    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        callbacks=[early_stopping],
        class_weight=class_weight,
        verbose=2,
    )

    print("Evaluation:")
    model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=1)

    y_proba = model.predict(x_test, batch_size=BATCH_SIZE).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))

    model.save(MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")

    FEATURE_COLUMNS_PATH.write_text(json.dumps(voltage_columns, indent=2))
    print(f"Saved feature columns to: {FEATURE_COLUMNS_PATH}")

    preprocessing_config = {
        "voltage_scaling": "per_window_min_max",
        "formula": "(x - window_min) / (window_max - window_min)",
        "constant_window_value": 0.0,
    }
    PREPROCESSING_CONFIG_PATH.write_text(json.dumps(preprocessing_config, indent=2))
    print(f"Saved preprocessing config to: {PREPROCESSING_CONFIG_PATH}")

    loaded_model = load_model(MODEL_PATH)
    print("Loaded saved model. Re-evaluating:")
    loaded_model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=1)


if __name__ == "__main__":
    main()
