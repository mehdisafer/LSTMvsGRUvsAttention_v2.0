# LSTMvsGRUvsAttention: Prédiction du prix du BTC (1 feature) avec LSTM pré-entraîné, GRU et Attention

Ce projet compare un LSTM pré-entraîné (5 couches, 64 neurones cachés) avec un GRU et un modèle à Attention définis et entraînés localement, pour prédire le prix du Bitcoin à partir de la **seule** colonne `Close` (input_dim=1). L’objectif est de comparer les performances de ces architectures sur 3 horizons (1 jour, 3 jours, 7 jours).

## Structure

```
LSTMvsGRUvsAttention/
├── data/
│   └── btc_prices.csv
├── LSTM/
│   ├── config.json
│   └── pytorch_model.bin
├── models/
│   ├── lstm_model.py
│   ├── gru_model.py
│   └── attention_model.py
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualization.py
├── notebooks/
│   └── comparaison_modeles.ipynb
├── requirements.txt
└── README.md
```

- **`data/btc_prices.csv`** : Données BTC journalières au moins sous la forme `[Date, Close]`.
- **`LSTM/config.json`** et **`LSTM/pytorch_model.bin`** : config et poids du modèle LSTM pré-entraîné.
- **`models/`** : définitions PyTorch des architectures LSTM, GRU, Attention.
- **`src/`** : Scripts Python pour prétraiter les données, entraîner/évaluer les modèles, etc.
- **`notebooks/`** : Notebook final où tout est exécuté et visualisé.

## Prérequis

- Python 3.11 ou supérieur

## Installation

1. **Cloner le dépôt :**

   ```bash
   git clone https://github.com/mehdisafer/LSTMvsGRUvsAttention_v2.0.git
   cd LSTMvsGRUvsAttention_v2.0
   ```

2. **Créer et activer un environnement virtuel :**

   ```bash
   python -m venv venv
   # Sous Linux/Mac :
   source venv/bin/activate
   # Sous Windows :
   venv\Scripts\activate
   ```

3. **Installer les dépendances :**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Placez votre dataset dans `data/btc_prices.csv`.
2. Placez la config (`config.json`) et les poids (`pytorch_model.bin`) du LSTM pré-entraîné dans `LSTM/`.
3. Ouvrez et exécutez le notebook `notebooks/comparaison_modeles.ipynb`.

## Points clés

- **input_dim=1** : on n’utilise que `Close` comme feature.
- **Fenêtre d’entrée** : par défaut 30 jours (fenêtre glissante).
- **Entraînement** : GRU et Attention sont entraînés sur 50 époques (avec early stopping).
- **Comparaison des métriques** : MAE, RMSE, MAPE, R², plus des graphiques comparatifs.
- **Comparaison visuelle** : 'comparaison_modeles.ipynb' offre des graphiques comparatifs.
- 
## Résultats

À la fin, vous obtiendrez :
- Les performances chiffrées des 3 modèles (LSTM pré-entraîné, GRU, Attention) sur 3 horizons (1, 3, 7 jours).
- Des graphiques comparant la courbe réelle et celles des prédictions.
