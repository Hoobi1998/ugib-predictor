#!/usr/bin/env python
# coding: utf-8

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from flask import Flask, render_template, request


APP_ROOT = Path(__file__).resolve().parent
MODEL_PATH = APP_ROOT / "ecg_1d_cnn.keras"
FEATURE_COLUMNS_PATH = APP_ROOT / "feature_columns.json"
PREPROCESSING_CONFIG_PATH = APP_ROOT / "preprocessing_config.json"
DATASET_CANDIDATES = (
    APP_ROOT / "ECG_window_sample.csv",
    APP_ROOT / "ECG_window_df.csv",
    APP_ROOT / "Desktop" / "ECG_window_df.csv",
    Path.home() / "Desktop" / "ECG_window_df.csv",
)
METADATA_COLUMNS = {"Window", "Patient", "Results", "Annotation"}
TEST_PATIENTS = {"233", "234"}
THRESHOLD = 0.5
MAX_SAMPLE_OPTIONS = 100


app = Flask(__name__)


def find_dataset_path() -> Optional[Path]:
    for path in DATASET_CANDIDATES:
        if path.exists():
            return path
    return None


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    unnamed_columns = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed_columns:
        df = df.drop(columns=unnamed_columns)
    return df


@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.exists():
        return None
    from tensorflow.keras.models import load_model

    return load_model(MODEL_PATH)


@lru_cache(maxsize=1)
def get_dataset() -> Optional[pd.DataFrame]:
    dataset_path = find_dataset_path()
    if dataset_path is None:
        return None
    return clean_dataframe(pd.read_csv(dataset_path))


def get_feature_count_from_metadata(dataset_path: Optional[Path]) -> int:
    if FEATURE_COLUMNS_PATH.exists():
        try:
            return len(json.loads(FEATURE_COLUMNS_PATH.read_text()))
        except json.JSONDecodeError:
            pass

    if dataset_path is None:
        return 0

    header = clean_dataframe(pd.read_csv(dataset_path, nrows=0))
    return len([col for col in header.columns if col not in METADATA_COLUMNS])


def get_sample_preview() -> Optional[pd.DataFrame]:
    dataset_path = find_dataset_path()
    if dataset_path is None:
        return None

    header = pd.read_csv(dataset_path, nrows=0)
    preview_columns = [col for col in ("Window", "Patient", "Results") if col in header.columns]
    if not preview_columns:
        return clean_dataframe(pd.read_csv(dataset_path, nrows=MAX_SAMPLE_OPTIONS))

    preview = clean_dataframe(pd.read_csv(dataset_path, usecols=preview_columns))
    if "Results" not in preview.columns:
        return preview.head(MAX_SAMPLE_OPTIONS)

    if "Patient" in preview.columns:
        patient_ids = preview["Patient"].astype(str)
        test_preview = preview.loc[patient_ids.isin(TEST_PATIENTS)]
        if not test_preview.empty:
            preview = test_preview

    per_class_limit = max(1, MAX_SAMPLE_OPTIONS // 2)
    normal = preview.loc[preview["Results"] == 0].head(per_class_limit)
    abnormal = preview.loc[preview["Results"] == 1].head(MAX_SAMPLE_OPTIONS - len(normal))
    balanced_preview = pd.concat([normal, abnormal]).sort_index()

    if len(balanced_preview) < MAX_SAMPLE_OPTIONS:
        remaining = preview.drop(index=balanced_preview.index).head(MAX_SAMPLE_OPTIONS - len(balanced_preview))
        balanced_preview = pd.concat([balanced_preview, remaining]).sort_index()

    return balanced_preview


def get_dataset_row(row_index: int) -> pd.DataFrame:
    dataset_path = find_dataset_path()
    if dataset_path is None:
        raise ValueError("Dataset was not found.")
    if row_index < 0:
        raise ValueError("Sample index must be non-negative.")

    row = pd.read_csv(
        dataset_path,
        skiprows=lambda index: index != 0 and index != row_index + 1,
        nrows=1,
    )
    if row.empty:
        raise ValueError(f"Sample index {row_index} was not found in the dataset.")
    return clean_dataframe(row)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    if FEATURE_COLUMNS_PATH.exists():
        columns = json.loads(FEATURE_COLUMNS_PATH.read_text())
        missing = [col for col in columns if col not in df.columns]
        if not missing:
            return columns

    return [
        col
        for col in df.columns
        if col not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
    ]


def normalize_uploaded_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_dataframe(df)

    if len(df.columns) == 1 and len(df.index) > 1:
        values = df.iloc[:, 0].to_numpy(dtype=np.float32)
        return pd.DataFrame([values], columns=[str(i) for i in range(len(values))])

    return df


def build_signal_payload(values: np.ndarray, limit: int = 900) -> list[float]:
    values = values.astype(float)
    if len(values) <= limit:
        return values.round(6).tolist()

    indices = np.linspace(0, len(values) - 1, limit).astype(int)
    return values[indices].round(6).tolist()


def min_max_scale_windows(windows: np.ndarray) -> np.ndarray:
    row_min = windows.min(axis=1, keepdims=True)
    row_max = windows.max(axis=1, keepdims=True)
    row_range = row_max - row_min
    safe_range = np.where(row_range == 0, 1.0, row_range)
    return (windows - row_min) / safe_range


def display_value(value):
    if pd.isna(value):
        return "לא ידוע"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def predict_rows(df: pd.DataFrame) -> dict:
    model = get_model()
    if model is None:
        raise RuntimeError("Model file ecg_1d_cnn.keras was not found. Train the model first.")

    feature_columns = get_feature_columns(df)
    if not feature_columns:
        raise ValueError("No numeric ECG voltage columns were found in the input.")

    x = df[feature_columns].to_numpy(dtype=np.float32)
    x = min_max_scale_windows(x)
    x = np.expand_dims(x, axis=-1)

    probabilities = model.predict(x, verbose=0).ravel()
    predictions = (probabilities >= THRESHOLD).astype(int)

    rows = []
    for row_index, probability in enumerate(probabilities):
        prediction = int(predictions[row_index])
        rows.append(
            {
                "row": row_index + 1,
                "probability": float(probability),
                "percent": round(float(probability) * 100, 2),
                "confidence": round((float(probability) if prediction == 1 else 1.0 - float(probability)) * 100, 2),
                "prediction": prediction,
                "label": "אק״ג לא תקין" if prediction == 1 else "אק״ג תקין",
                "class_name": "abnormal" if prediction == 1 else "normal",
            }
        )

    return {
        "rows": rows,
        "signal": build_signal_payload(df.iloc[0][feature_columns].to_numpy(dtype=np.float32)),
        "feature_count": len(feature_columns),
    }


def sample_options(dataset: Optional[pd.DataFrame]) -> list[dict]:
    if dataset is None:
        return []

    options = []
    for index in dataset.index:
        patient = dataset.at[index, "Patient"] if "Patient" in dataset.columns else "לא ידוע"
        window = dataset.at[index, "Window"] if "Window" in dataset.columns else index
        result = dataset.at[index, "Results"] if "Results" in dataset.columns else None
        label = "לא תקין" if result == 1 else "תקין" if result == 0 else "לא ידוע"
        options.append(
            {
                "index": int(index),
                "text": f"מטופל {display_value(patient)} | חלון {display_value(window)} | {label}",
            }
        )
    return options


def status_payload() -> dict:
    dataset_path = find_dataset_path()
    return {
        "model_ready": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "dataset_ready": dataset_path is not None,
        "dataset_path": str(dataset_path) if dataset_path else "",
        "feature_count": get_feature_count_from_metadata(dataset_path),
    }


@app.route("/", methods=["GET"])
def index():
    sample_preview = get_sample_preview()
    return render_template(
        "index.html",
        status=status_payload(),
        samples=sample_options(sample_preview),
        selected_source="sample",
        selected_sample_index=None,
        result=None,
        error=None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    sample_preview = get_sample_preview()
    result = None
    error = None
    source = request.form.get("source", "sample")
    selected_sample_index = request.form.get("sample_index")

    try:
        if source == "sample":
            sample_index = int(selected_sample_index)
            input_df = get_dataset_row(sample_index)
            result = predict_rows(input_df)
            true_label = input_df.iloc[0].get("Results", None)
            if true_label in (0, 1):
                result["true_label"] = "אק״ג לא תקין" if true_label == 1 else "אק״ג תקין"
        else:
            upload = request.files.get("ecg_file")
            if upload is None or upload.filename == "":
                raise ValueError("Choose a CSV file or select a dataset sample.")
            input_df = normalize_uploaded_dataframe(pd.read_csv(upload))
            result = predict_rows(input_df)
    except Exception as exc:
        error = str(exc)

    return render_template(
        "index.html",
        status=status_payload(),
        samples=sample_options(sample_preview),
        selected_source=source,
        selected_sample_index=selected_sample_index,
        result=result,
        error=error,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
