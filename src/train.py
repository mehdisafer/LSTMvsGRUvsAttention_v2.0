"""Device-consistent training with explicit validation of usable batches."""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from numbers import Integral
from typing import Any

import torch
from torch import nn, optim
from torch.utils.data import DataLoader


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _checked_loss(output: torch.Tensor, target: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    if output.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {output.shape} != {target.shape}")
    loss = criterion(output, target)
    if not torch.isfinite(loss).item():
        raise ValueError("Non-finite loss; check inputs, targets and model output")
    return loss


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Mapping[str, Any],
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, list[float]]]:
    """Train and restore the best validation weights, preserving the original API.

    Both loaders may yield CPU tensors, even when training on CUDA. Loss means
    use the number of target elements actually yielded (including with drop_last
    or a sampler), matching MSELoss rather than the underlying dataset size. Empty loaders, invalid
    configuration, shape broadcasting and non-finite losses fail explicitly.
    """
    epochs = _positive_integer(config.get("epochs", 50), "epochs")
    patience = _positive_integer(config.get("patience", 5), "patience")
    learning_rate = float(config.get("lr", 1e-3))
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("lr must be finite and greater than zero")

    # Move the model before creating the optimizer; all batches follow it.
    device = torch.device(device)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float("inf")
    best_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        train_count = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if inputs.size(0) == 0:
                continue
            count = targets.numel()
            optimizer.zero_grad()
            loss = _checked_loss(model(inputs), targets, criterion)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * count
            train_count += count
        if train_count == 0:
            raise ValueError("train_loader yielded no examples")

        model.eval()
        running_val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                if inputs.size(0) == 0:
                    continue
                count = targets.numel()
                loss = _checked_loss(model(inputs), targets, criterion)
                running_val_loss += loss.item() * count
                val_count += count
        if val_count == 0:
            raise ValueError("val_loader yielded no examples")

        train_loss = running_train_loss / train_count
        val_loss = running_val_loss / val_count
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"[Epoch {epoch}/{epochs}] train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping déclenché.")
                break
    model.load_state_dict(best_wts)
    return model, history
