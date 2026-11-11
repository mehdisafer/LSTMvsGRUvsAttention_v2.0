"""Additional boundary cases found while reviewing the unpublished candidate."""
import math

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset


@pytest.mark.parametrize("header", ["date,close,close", "date,date,close", '"date","close","close"'])
def test_exact_duplicate_csv_headers_are_rejected(pipeline, tmp_path, header):
    path = tmp_path / "ambiguous.csv"
    path.write_text(header + "\n2026-01-01,1,2\n", encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Duplicate"):
        pipeline.preprocess.load_and_filter_data(path)
    assert path.read_bytes() == before


def test_csv_bom_blank_lines_and_missing_row_policy(pipeline, tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text("\ufeff\nDate,Close\n2026-01-03,12\n2026-01-02,\n2026-01-01,10\n", encoding="utf-8")
    before = path.read_bytes()
    frame = pipeline.preprocess.load_and_filter_data(path)
    assert frame.close.tolist() == [10, 12]
    assert path.read_bytes() == before


class ZeroModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, inputs):
        return inputs * self.weight * 0


def test_loss_weights_actual_target_elements(pipeline):
    # Both batches have one example, but contain 1 and 3 MSE elements.
    # Their means are 1 and 4, so the full element mean is 13/4, not 5/2.
    batches = [(torch.ones(1, 1), torch.ones(1, 1)),
               (torch.ones(1, 3), torch.full((1, 3), 2.0))]
    data = DataLoader(batches, batch_size=None)
    _, history = pipeline.train.train_model(ZeroModel(), data, data, {"epochs": 1})
    assert history["train_loss"] == [3.25]
    assert history["val_loss"] == [3.25]


def test_loss_weights_sampler_not_dataset_length(pipeline):
    dataset = TensorDataset(torch.ones(5, 1), torch.ones(5, 1))
    data = DataLoader(dataset, batch_size=2, sampler=SubsetRandomSampler([0, 4]))
    _, history = pipeline.train.train_model(ZeroModel(), data, data, {"epochs": 1})
    assert history["train_loss"] == [1.0]
    assert history["val_loss"] == [1.0]


@pytest.mark.parametrize("predictions", [np.array([2.0, 2.0]), np.array([1.0, 3.0])])
def test_constant_target_r2_is_explicitly_undefined(pipeline, predictions):
    metrics = pipeline.evaluate.compute_metrics(predictions, np.array([2.0, 2.0]))
    assert math.isnan(metrics["R2"])
    assert math.isfinite(metrics["MAE"])
    assert math.isfinite(metrics["RMSE"])


def test_csv_string_missing_values_keep_original_drop_policy(pipeline, tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text("Date,Close\n2026-01-01,NA\n2026-01-02,11\nNULL,12\n", encoding="utf-8")
    frame = pipeline.preprocess.load_and_filter_data(path)
    assert frame.close.tolist() == [11]
