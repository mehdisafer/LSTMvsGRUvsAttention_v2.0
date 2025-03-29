import matplotlib.pyplot as plt
import seaborn as sns

def plot_loss(history, title="Loss curve"):
    plt.figure()
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'],   label='Val')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title(title)
    plt.legend()
    plt.show()

def plot_predictions_compare(dates, real_vals, pred_gru, pred_att, horizon):
    """
    Affiche sur un même graphe: Réel vs GRU vs Attention
    """
    plt.figure(figsize=(10,4))
    plt.plot(dates, real_vals, label="Réel")
    plt.plot(dates, pred_gru,  label=f"GRU(h={horizon})")
    plt.plot(dates, pred_att,  label=f"Attention(h={horizon})")
    plt.xlabel("Date")
    plt.ylabel("Prix")
    plt.title(f"Comparaison Réel vs GRU vs Attention (h={horizon})")
    plt.legend()
    plt.show()

def plot_prediction_lstm(dates, real_vals, pred_lstm, horizon):
    """
    Graph pour LSTM pré-entraîné
    """
    plt.figure(figsize=(10,4))
    plt.plot(dates, real_vals,   label="Réel")
    plt.plot(dates, pred_lstm,   label=f"LSTM(h={horizon})")
    plt.xlabel("Date")
    plt.ylabel("Prix")
    plt.title(f"Comparaison Réel vs LSTM (h={horizon})")
    plt.legend()
    plt.show()

def barplot_metrics(df_results):
    """
    df_results => [Model, Horizon, MAE, RMSE, MAPE, R2]
    """
    metrics_list = ["MAE","RMSE","MAPE","R2"]
    for m in metrics_list:
        plt.figure()
        sns.barplot(data=df_results, x="Horizon", y=m, hue="Model")
        plt.title(f"Comparaison {m}")
        plt.legend()
        plt.show()
