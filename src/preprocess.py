import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

def load_and_filter_data(csv_path):
    """
    Charge le CSV, conserve seulement [Date, Close].
    """
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df.rename(columns=str.lower)
    
    # Garde seulement date, close
    df = df[['date','close']].dropna()
    df = df.sort_values(by='date')
    df.set_index('date', inplace=True)
    
    return df

def train_val_test_split(df, train_ratio=0.8, val_ratio=0.1):
    """
    Sépare df en train/val/test selon les ratios (chronologique).
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))
    
    df_train = df.iloc[:train_end]
    df_val   = df.iloc[train_end:val_end]
    df_test  = df.iloc[val_end:]
    return df_train, df_val, df_test

def create_sequences(series, window_size=30, horizon=1):
    """
    series: array (N,) => la série 'Close' univariée.
    """
    X, y = [], []
    for i in range(len(series) - window_size - horizon + 1):
        seq_x = series[i : i+window_size]        # shape (window_size,)
        seq_y = series[i + window_size + horizon - 1]
        X.append(seq_x)
        y.append(seq_y)
    
    X = np.array(X).reshape(-1, window_size, 1)  # (samples, seq_len=window_size, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y

def prepare_datasets(df_train, df_val, df_test, window_size=30, horizon=1):
    """
    Applique MinMaxScaler sur la col 'close' du TRAIN, puis crée X,y (univarié).
    """
    scaler = MinMaxScaler()
    
    train_vals = df_train['close'].values.reshape(-1,1)
    val_vals   = df_val['close'].values.reshape(-1,1)
    test_vals  = df_test['close'].values.reshape(-1,1)
    
    scaler.fit(train_vals)  # fit sur train
    
    train_scaled = scaler.transform(train_vals).flatten()
    val_scaled   = scaler.transform(val_vals).flatten()
    test_scaled  = scaler.transform(test_vals).flatten()
    
    X_train, y_train = create_sequences(train_scaled, window_size, horizon)
    X_val,   y_val   = create_sequences(val_scaled,   window_size, horizon)
    X_test,  y_test  = create_sequences(test_scaled,  window_size, horizon)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler

def numpy_to_torch(X, y):
    """
    Convertit X,y -> tenseurs float32. 
    X shape: (batch_size, seq_len=window_size, 1)
    y shape: (batch_size, 1)
    """
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()
    return X_t, y_t
