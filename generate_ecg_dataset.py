#!/usr/bin/env python
# coding: utf-8

from pathlib import Path

import numpy as np
import pandas as pd
import wfdb


OUTPUT_PATH = Path("ECG_window_df.csv")
DATABASE = "mitdb"
CHANNEL = 0
SAMPLING_RATE = 360
MAX_SAMPLES_PER_RECORD = 650000
WINDOW_SECONDS = 2
WINDOW_SIZE = SAMPLING_RATE * WINDOW_SECONDS
NORMAL_SYMBOLS = {"N", "+", "~", "|"}


def label_windows(record_name: str, total_samples: int) -> np.ndarray:
    annotations = wfdb.rdann(record_name, "atr", pn_dir=DATABASE, sampto=total_samples)
    labels = np.zeros(total_samples // WINDOW_SIZE, dtype=np.int8)

    for sample, symbol in zip(annotations.sample, annotations.symbol):
        window_index = sample // WINDOW_SIZE
        if window_index >= len(labels):
            continue
        if symbol not in NORMAL_SYMBOLS:
            labels[window_index] = 1

    return labels


def build_record_windows(record_name: str) -> pd.DataFrame:
    signal, _ = wfdb.rdsamp(
        record_name,
        pn_dir=DATABASE,
        channels=[CHANNEL],
        sampto=MAX_SAMPLES_PER_RECORD,
    )
    voltages = signal[:, 0].astype(np.float32)

    usable_samples = (len(voltages) // WINDOW_SIZE) * WINDOW_SIZE
    voltages = voltages[:usable_samples]
    windows = voltages.reshape(-1, WINDOW_SIZE)
    labels = label_windows(record_name, usable_samples)

    df = pd.DataFrame(windows, columns=[str(i) for i in range(WINDOW_SIZE)])
    df.insert(0, "Results", labels)
    df.insert(0, "Patient", record_name)
    df.insert(0, "Window", np.arange(len(df), dtype=np.int32))

    return df


def main() -> None:
    records = wfdb.io.get_record_list(db_dir=DATABASE, records="all")
    print(f"Found {len(records)} records in {DATABASE}.", flush=True)
    print(f"Window size: {WINDOW_SIZE} samples ({WINDOW_SECONDS} seconds).", flush=True)

    frames = []
    for position, record_name in enumerate(records, start=1):
        print(f"[{position}/{len(records)}] Processing record {record_name}", flush=True)
        frames.append(build_record_windows(record_name))

    dataset = pd.concat(frames, ignore_index=True)
    dataset.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved dataset to: {OUTPUT_PATH}", flush=True)
    print(f"Dataset shape: {dataset.shape}", flush=True)
    print("Class distribution:", flush=True)
    print(dataset["Results"].value_counts().sort_index(), flush=True)


if __name__ == "__main__":
    main()
