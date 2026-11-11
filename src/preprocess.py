"""Chronological, train-only normalization for the univariate benchmark."""
from __future__ import annotations

from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


def load_and_filter_data(csv_path: str | Path) -> pd.DataFrame:
    """Read Date/Close (case-insensitive); discard missing rows as before.

    Normalize headers *before* parsing dates. Invalid values and ambiguous
    duplicate timestamps fail rather than silently changing the time series.
    """
    # Read the header as data so pandas cannot silently rename an exact
    # duplicate (close,close -> close,close.1) before validation.
    raw = pd.read_csv(csv_path, header=None, dtype=str)
    if raw.empty:
        raise ValueError("CSV contains no header")
    columns = [str(column).strip().lower() for column in raw.iloc[0]]
    df = raw.iloc[1:].copy()
    df.columns = columns
    if not df.columns.is_unique:
        raise ValueError("Duplicate CSV columns after header normalization")
    missing = {"date", "close"}.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")
    df = df[["date", "close"]].dropna().copy()
    if df.empty:
        raise ValueError("CSV contains no usable Date/Close rows")
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["close"] = pd.to_numeric(df["close"], errors="raise")
    if df["date"].isna().any() or not np.isfinite(df["close"].to_numpy(dtype=float)).all():
        raise ValueError("Date/Close contains a missing date or non-finite value")
    if df["date"].duplicated().any():
        raise ValueError("Duplicate timestamps require explicit source reconciliation")
    return df.sort_values("date").set_index("date")


def train_val_test_split(
    df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create three non-empty, ordered partitions with no overlapping dates."""
    if (
        not np.isfinite([train_ratio, val_ratio]).all()
        or train_ratio <= 0
        or val_ratio <= 0
        or train_ratio + val_ratio >= 1
    ):
        raise ValueError("Ratios must be positive, finite and sum to less than one")
    if not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise ValueError("Input index must be unique and chronologically sorted")
    train_end = int(len(df) * train_ratio)
    val_end = int(len(df) * (train_ratio + val_ratio))
    if not 0 < train_end < val_end < len(df):
        raise ValueError("Ratios and sample count must produce three non-empty partitions")
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def create_sequences(
    series: np.ndarray, window_size: int = 30, horizon: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Return X=(N, window, 1), y=(N, 1), predicting the selected future horizon."""
    for name, value in (("window_size", window_size), ("horizon", horizon)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    values = np.asarray(series, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("series must be a finite one-dimensional numeric array")
    sample_count = len(values) - window_size - horizon + 1
    if sample_count < 1:
        raise ValueError(f"Need at least {window_size + horizon} observations for this window/horizon")
    inputs = np.stack([values[i:i + window_size] for i in range(sample_count)])
    targets = values[window_size + horizon - 1:]
    return inputs.reshape(-1, window_size, 1), targets.reshape(-1, 1)


def prepare_datasets(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    window_size: int = 30,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Fit scaling on training observations only; keep sequences within each split.

    Validation/test values may exceed [0, 1]. They are not clipped or refitted.
    Each split needs at least window_size + horizon observations.
    """
    frames = (df_train, df_val, df_test)
    values = []
    for name, frame in zip(("train", "validation", "test"), frames):
        if frame.empty or "close" not in frame:
            raise ValueError(f"{name} partition must contain close observations")
        if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
            raise ValueError(f"{name} partition index must be sorted and unique")
        array = frame["close"].to_numpy(dtype=float).reshape(-1, 1)
        if not np.isfinite(array).all():
            raise ValueError(f"{name} partition contains non-finite close values")
        values.append(array)
    if not (df_train.index[-1] < df_val.index[0] and df_val.index[-1] < df_test.index[0]):
        raise ValueError("Partitions overlap or are not in train/validation/test order")
    scaler = MinMaxScaler().fit(values[0])
    sequences = [create_sequences(scaler.transform(value).ravel(), window_size, horizon) for value in values]
    return (*sequences[0], *sequences[1], *sequences[2], scaler)


def numpy_to_torch(X: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert the existing sequence contract to float32 tensors."""
    return torch.from_numpy(X).float(), torch.from_numpy(y).float()
