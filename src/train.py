import torch
import torch.nn as nn
import torch.optim as optim
import copy

def train_model(model, train_loader, val_loader, config, device='cpu'):
    """
    Entraîne un modèle (LSTM/GRU/Attention).
    config: {'lr':..., 'epochs':..., 'patience':...}
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))
    
    epochs   = config.get("epochs", 50)
    patience = config.get("patience", 5)
    
    best_val_loss = float('inf')
    best_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    for epoch in range(1, epochs+1):
        model.train()
        running_train_loss = 0.0
        
        for Xb, Yb in train_loader:
            
            optimizer.zero_grad()
            out = model(Xb)           # shape: (batch,1)
            loss = criterion(out, Yb) # MSE
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * Xb.size(0)
        
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for Xv, Yv in val_loader:
                Xv, Yv = Xv.to(device), Yv.to(device)
                preds = model(Xv)
                val_loss = criterion(preds, Yv)
                running_val_loss += val_loss.item() * Xv.size(0)
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        
        print(f"[Epoch {epoch}/{epochs}] train_loss={epoch_train_loss:.6f}, val_loss={epoch_val_loss:.6f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping déclenché.")
                break
    
    model.load_state_dict(best_wts)
    return model, history
