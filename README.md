# LSTM vs GRU vs Attention — benchmark univarié

Ce projet compare trois architectures PyTorch à partir de la colonne `Close`, avec des horizons de 1, 3 et 7 observations. Pour une série quotidienne sans trou, ces horizons correspondent à 1, 3 et 7 jours. Une ligne manquante n'est pas automatiquement interpolée : le pas reste une observation, pas nécessairement un jour calendaire.

Le notebook historique compare notamment un LSTM pré-entraîné à un GRU et à une architecture LSTM + attention. Les tests et le parcours synthétique ajoutés ici **ne chargent pas les poids pré-entraînés** et ne constituent pas une validation des performances sur le marché BTC.

## Structure réelle du dépôt

```text
comparaison_modeles.ipynb       # Notebook à la racine, conservé
LSTM/config.json               # Configuration historique, conservée
LSTM/pytorch_model.bin          # Poids historiques, conservés
models/                        # LSTM, GRU, Attention : architectures conservées
src/preprocess.py              # Lecture, séparation temporelle et séquences
src/train.py                   # Entraînement, validation et meilleur checkpoint
src/evaluate.py                # Prédiction et métriques
src/visualization.py            # Visualisation historique, conservée
data/BTC.csv                   # Dataset historique, conservé
scripts/smoke_benchmark.py      # Parcours CPU synthétique et déterministe
requirements-test.txt          # Dépendances directes de test épinglées
.gitattributes                  # Octets LF identiques dans les checkouts CI
pytest.ini
tests/
.github/workflows/tests.yml     # CI CPU Linux/Windows
```

## Installation de test CPU

La cible de la matrice CI est Python 3.12 et 3.13. Le candidat courant est testé localement sous **Python 3.13.5 / Linux / PyTorch 2.10.0+cpu**. La première révision publiée a aussi exécuté les tests CPU sur les deux systèmes ; la correction de provenance ajoutée ensuite doit être requalifiée sur sa propre révision. CUDA reste non vérifié.

Dans un environnement virtuel isolé :

```bash
python -m venv .venv
# Linux/macOS : source .venv/bin/activate
# PowerShell : .\.venv\Scripts\Activate.ps1
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
python -m pytest --junitxml=test-results.xml
python scripts/smoke_benchmark.py --output smoke-results.json
```

Le fichier historique `requirements.txt` est préservé. Il contient des dépendances plus larges et non verrouillées : il n'est pas utilisé pour la recette CPU. Les dépendances directes de test sont épinglées, mais ce n'est pas un lockfile transitif complet.

Pour le notebook et les visualisations historiques, installer les dépendances nécessaires à ses cellules dans un environnement distinct. Le notebook et ses résultats historiques n'ont pas été réexécutés dans cette qualification.

## Contrats des données

`load_and_filter_data()` accepte les en-têtes `Date,Close` ou `date,close`, ainsi que les espaces autour des noms. La normalisation précède l'analyse des dates. Les lignes manquantes sont retirées comme dans le parcours historique. Les valeurs invalides, infinies, les noms de colonnes ambigus (y compris les doublons strictement identiques avant renommage par pandas) et les timestamps dupliqués sont refusés explicitement.

`train_val_test_split()` conserve une séparation chronologique, sans mélange aléatoire. Les ratios doivent produire trois partitions non vides et disjointes.

`prepare_datasets()` ajuste le `MinMaxScaler` **uniquement sur le train**. Les valeurs de validation et de test peuvent dépasser `[0,1]` ; elles ne sont ni recalées ni tronquées. Les séquences restent à l'intérieur de leur partition, comme dans l'implémentation historique. Chaque partition nécessite au moins `window_size + horizon` observations. Cette politique perd les premières fenêtres de chaque partition ; elle n'est pas un protocole walk-forward complet.

## Entraînement et placement CPU/GPU

Les signatures des fonctions existantes sont conservées. Le modèle est déplacé vers `device` avant création de l'optimiseur, puis les entrées **et les cibles** d'entraînement, de validation et de prédiction sont déplacées vers le même appareil.

```python
model, history = train_model(model, train_loader, val_loader,
                             {"lr": 1e-3, "epochs": 50, "patience": 5},
                             device="cuda")
```

Cet exemple nécessite une installation PyTorch CUDA adaptée et un GPU disponible. Le test CUDA est conditionnel ; un résultat `skipped` n'est pas une validation GPU.

Les pertes moyennes portent sur les éléments cibles réellement traités, conformément à `MSELoss(reduction="mean")`, et non sur la taille brute du dataset. Dans ce benchmark univarié, un élément cible correspond à un exemple. La pondération reste correcte avec `drop_last`, un sampler ou des lots de tailles de sortie différentes. Les loaders vides, les formes incompatibles et les pertes non finies sont refusés. Le meilleur état de validation est restauré à la fin ; aucune preuve de qualité n'est déduite du simple achèvement de l'entraînement.

## Évaluation

Les métriques restent `MAE`, `RMSE`, `MAPE` et `R2`. Les calculer dans l'unité d'origine :

```python
scaled_preds, scaled_truth = predict(model, test_loader, device="cpu")
metrics = compute_metrics(scaler.inverse_transform(scaled_preds),
                          scaler.inverse_transform(scaled_truth))
```

`MAPE` est `NaN` lorsqu'une valeur réelle vaut zéro, au lieu d'utiliser un dénominateur artificiel. `R2` est `NaN` pour une seule observation ou une série cible constante. Ce choix explicite évite les valeurs 0/1 de remplacement de scikit-learn sur cible constante ; ce n'est pas une nouvelle mesure de précision. Les vecteurs vides, de longueurs différentes, non finis ou multivariés sont refusés. Ces refus sont des changements explicites par rapport aux entrées auparavant non contrôlées.

## Recette reproductible

Le script synthétique utilise les trois classes du dépôt, les horizons 1/3/7, un petit modèle réinitialisé pour chaque essai, une graine fixe, un thread CPU et des algorithmes déterministes. Il compare chaque sortie à une référence de persistance : la dernière valeur observée de la fenêtre.

Le rapport JSON contient les paramètres, les versions de l'environnement, les métriques, la référence naïve et les empreintes du code, des tests, du workflow, des dépendances directes et des données. Les chemins du manifeste utilisent `/` sur tous les systèmes et le CSV synthétique est sérialisé en LF. Les empreintes portent sur les octets réels, sans normalisation qui masquerait une différence. En CI, il ajoute le SHA checkout, le SHA de tête de PR et les identifiants du run ; hors GitHub, ces identifiants restent nuls. Deux processus indépendants ont produit des rapports identiques dans l'environnement local testé. Cela ne promet pas une égalité bit-à-bit entre systèmes, appareils ou versions de PyTorch.

Le contrôle `--verify-git`, activé automatiquement dans GitHub Actions, compare les objets Git calculés sur ces octets au checkout effectivement testé. Il refuse un fichier source modifié ou non suivi et un SHA de checkout différent de celui du run. Les attributs Git imposent LF aux sources et configurations sans changer les fichiers historiques binaires, CSV et notebook. Hors checkout, la recette peut fonctionner sans ce contrôle mais inscrit explicitement `NOT_RUN` dans le rapport.

Aucun score synthétique ne permet de conclure qu'un modèle est supérieur sur des données réelles. Pour cela, il reste à construire une évaluation temporelle walk-forward, documenter l'origine des poids pré-entraînés et réserver une période finale de test non utilisée pour sélectionner les modèles.

## CI et niveau de preuve

Le workflow candidat cible Ubuntu et Windows, Python 3.12/3.13, avec permissions en lecture et actions épinglées par commit. Il conserve JUnit et le rapport synthétique. Il n'utilise aucun secret et ne déploie rien.

**Qualification locale du 11 novembre 2026 : 68 tests réussis, 1 test CUDA ignoré ; compilation réussie ; 9 cas synthétiques exécutés deux fois. La première CI a révélé une conversion LF/CRLF des sources sous Windows malgré des tests réussis. La correction de provenance est testée séparément et ses résultats hébergés doivent être consultés sur la révision exacte de la PR ; les résultats du parent ne lui sont pas transférés.** Voir [le dossier de validation](docs/VALIDATION.md).
