import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def predict(model, data_loader, device='cpu'):
    """
    Retourne (preds, trues) sur l'ensemble du data_loader.
    """
    model.eval()
    preds, trues = [], []
    
    with torch.no_grad():
        for Xb, Yb in data_loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            out = model(Xb)
            preds.append(out.cpu().numpy())
            trues.append(Yb.cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return preds, trues

def compute_metrics(preds, trues):
    """
    Calcule MAE, RMSE, MAPE, R2.
    preds, trues => (N,1) ou (N,)
    """
    preds = preds.flatten()
    trues = trues.flatten()
    
    mae = mean_absolute_error(trues, preds)
    mse = mean_squared_error(trues, preds)
    rmse = np.sqrt(mse)
    eps = 1e-9
    mape = np.mean(np.abs((trues - preds)/(trues+eps))) * 100
    r2 = r2_score(trues, preds)
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }
