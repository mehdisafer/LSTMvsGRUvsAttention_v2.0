"""Prediction and metrics with explicit undefined/invalid-data handling."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader


def predict(
    model: nn.Module, data_loader: DataLoader, device: str | torch.device = "cpu"
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate on the requested device; return predictions and truths on CPU."""
    model.to(torch.device(device))
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if inputs.size(0) == 0:
                continue
            output = model(inputs)
            if output.shape != targets.shape:
                raise ValueError(f"Prediction/target shape mismatch: {output.shape} != {targets.shape}")
            if not torch.isfinite(output).all().item() or not torch.isfinite(targets).all().item():
                raise ValueError("Predictions and targets must be finite")
            preds.append(output.cpu().numpy())
            trues.append(targets.cpu().numpy())
    if not preds:
        raise ValueError("data_loader yielded no examples")
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def _vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim not in (1, 2) or (array.ndim == 2 and array.shape[1] != 1):
        raise ValueError(f"{name} must have shape (N,) or (N, 1)")
    array = array.ravel()
    if not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite observations")
    return array


def compute_metrics(preds: np.ndarray, trues: np.ndarray) -> dict[str, float]:
    """Return the original four metrics on equally-sized finite vectors.

    Supply original-unit values (inverse-transform scaled outputs first).
    MAPE is NaN if any actual value is zero; no artificial epsilon denominator
    is substituted. R2 is NaN for fewer than two observations or a constant target series.
    """
    preds, trues = _vector(preds, "preds"), _vector(trues, "trues")
    if len(preds) != len(trues):
        raise ValueError("preds and trues must contain the same number of observations")
    return {
        "MAE": float(mean_absolute_error(trues, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(trues, preds))),
        "MAPE": float(np.mean(np.abs((trues - preds) / trues)) * 100) if np.all(trues != 0) else float("nan"),
        "R2": float(r2_score(trues, preds, force_finite=False))
        if len(trues) > 1 and not np.all(trues == trues[0]) else float("nan"),
    }
