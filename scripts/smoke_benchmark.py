"""Exercise all three architectures on synthetic data; not a market benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from importlib.metadata import version
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.attention_model import AttentionModel
from models.gru_model import GRUModel
from models.lstm_model import LSTMModel
from src.evaluate import compute_metrics, predict
from src.preprocess import numpy_to_torch, prepare_datasets, train_val_test_split
from src.train import train_model


def run_smoke(seed: int = 2026) -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dates = pd.date_range('2025-01-01', periods=180, freq='D')
    time = np.arange(len(dates), dtype=float)
    # No network, no checkpoint loading and no external dataset.
    values = 100 + .05 * time + np.sin(time / 7)
    frame = pd.DataFrame({'close': values}, index=dates)
    train, validation, test = train_val_test_split(frame, .6, .2)
    results = []
    for horizon in (1, 3, 7):
        X, y, Xv, yv, Xt, yt, scaler = prepare_datasets(train, validation, test, 8, horizon)
        training_loader = DataLoader(TensorDataset(*numpy_to_torch(X, y)), batch_size=16)
        validation_loader = DataLoader(TensorDataset(*numpy_to_torch(Xv, yv)), batch_size=16)
        test_loader = DataLoader(TensorDataset(*numpy_to_torch(Xt, yt)), batch_size=16)
        actuals = scaler.inverse_transform(yt)
        persistence = scaler.inverse_transform(Xt[:, -1, :])
        baseline = compute_metrics(persistence, actuals)
        for name, model_type in [('LSTM', LSTMModel), ('GRU', GRUModel), ('Attention', AttentionModel)]:
            torch.manual_seed(seed)
            model = model_type(input_dim=1, hidden_dim=4, num_layers=1, output_dim=1)
            model, history = train_model(model, training_loader, validation_loader,
                                         {'epochs': 2, 'patience': 2}, device='cpu')
            scaled_predictions, scaled_truth = predict(model, test_loader, 'cpu')
            predictions = scaler.inverse_transform(scaled_predictions)
            truth = scaler.inverse_transform(scaled_truth)
            metrics = compute_metrics(predictions, truth)
            if not all(np.isfinite(list(metrics.values()))):
                raise RuntimeError(f'Non-finite synthetic metrics for {name}/{horizon}')
            results.append({'architecture': name, 'horizon': horizon,
                            'test_examples': len(truth), 'metrics': metrics,
                            'persistence_baseline': baseline, 'history': history})
    source_paths = sorted(
        [p for directory in ("src", "models", "tests", "scripts")
         for p in (ROOT / directory).rglob("*.py")]
        + [ROOT / "requirements-test.txt", ROOT / "pytest.ini",
           ROOT / ".github/workflows/tests.yml"]
    )
    return {
        'scope': 'CPU synthetic integration smoke; not historical BTC accuracy',
        'pretrained_checkpoint_loaded': False,
        'seed': seed,
        'python': platform.python_version(),
        'platform': platform.system(),
        'torch': torch.__version__, 'numpy': np.__version__, 'pandas': pd.__version__,
        'sklearn': version('scikit-learn'), 'pytest': version('pytest'),
        'github': {'checkout_sha': os.getenv('GITHUB_SHA'),
                   'source_head_sha': os.getenv('SOURCE_HEAD_SHA'),
                   'run_id': os.getenv('GITHUB_RUN_ID'),
                   'run_attempt': os.getenv('GITHUB_RUN_ATTEMPT'),
                   'event': os.getenv('GITHUB_EVENT_NAME')},
        'cuda_exercised': False, 'threads': 1,
        'dataset_sha256': hashlib.sha256(frame.to_csv().encode()).hexdigest(),
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in source_paths},
        'results': results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('smoke-results.json'))
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error('--seed must be between 0 and 2**32 - 1')
    result = run_smoke(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Synthetic smoke completed: {len(result["results"])} model/horizon cases')


if __name__ == '__main__':
    main()
