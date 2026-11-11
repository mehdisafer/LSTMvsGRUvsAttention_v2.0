# Qualification du candidat — 11 novembre 2026

## Origine et continuité

Dépôt : `mehdisafer/LSTMvsGRUvsAttention_v2.0`.
Baseline Git relue : `2630b916b114e128a722f42b9b609348f8f4111a`.
Le premier dossier local contenait 55 tests réussis et 1 test CUDA ignoré. Cette reprise a reproduit ce résultat, puis ajouté 9 tests de revue. Six de ces nouveaux cas échouaient sur le premier candidat : trois variantes de doublons d'en-têtes masqués par pandas, une moyenne MSE pondérée par exemples plutôt que par éléments cibles, et deux cas de R2 sur cible constante. Les trois autres cas renforcent la couverture des samplers et de la politique CSV existante.

Les huit blobs de la première matérialisation sont vérifiés par OID Git ; `src/visualization.py` a ensuite été relu et vérifié à l'identique (`8f69d046fc31c5196b7d6303586274c73912f969`). Le notebook, les poids et le dataset historique restent inchangés. Le checkout local de qualification est une matérialisation partielle, pas un clone complet. Le consommateur de visualisation a été lu, sans être exécuté. Le contenu du notebook n'a pas été récupéré par la lecture disponible.

## Résultats locaux observés

| Contrôle | Résultat | Portée |
|---|---|---|
| Premier candidat retrouvé | 55 réussites, 1 test ignoré | Python 3.13.5 / Linux / CPU |
| Neuf tests de revue sur ce premier candidat | 6 échecs, 3 réussites | Défauts/contrats explicités ci-dessus |
| Candidat final | 64 réussites, 1 test ignoré | Même environnement CPU |
| Suite finale sur les sources originales | 43 échecs, 21 réussites, 1 test ignoré | Inclut de nouveaux contrats ; pas 43 bugs distincts |
| Compilation | Réussie | Sources, modèles, tests et script |
| Parcours synthétique | 9 couples architecture/horizon, deux fois | Trois architectures du dépôt, 1/3/7 observations |
| Répétabilité locale | Rapports JSON identiques entre les deux processus | Même environnement uniquement |
| CUDA | Non exécuté, test ignoré | Aucun GPU disponible |
| Notebook / poids pré-entraînés | Non exécutés / non chargés | Aucun score BTC historique requalifié |

Les commandes reproductibles figurent dans le README. Le test de transfert d'appareil sur CPU instrumente les appels réels à `Tensor.to`. Il démontre les appels train/validation et l'ordre modèle/optimiseur, pas une exécution CUDA. Le test matériel séparé reste ignoré sans GPU.

## Contrats et compatibilité

Les noms d'arguments, valeurs par défaut et clés de sortie des fonctions publiques sont conservés. Les données invalides lèvent désormais des erreurs explicites. Les en-têtes sont lus avant tout renommage automatique des doublons. Le scaler reste ajusté au train uniquement ; les séquences restent internes à chaque partition.

La moyenne d'époque est pondérée par les éléments cibles effectivement traités, conformément à la réduction MSE. Les anciens résultats biaisés avec `drop_last` ou un sampler changent. `MAPE` est NaN lorsqu'une valeur vraie vaut zéro. La politique du benchmark représente par NaN le R2 sans variation de cible (ou avec un seul exemple) au lieu des valeurs de remplacement 0/1 de scikit-learn. Les autres métriques restent calculées. Les modèles et poids historiques ne sont pas modifiés.

## CI et conditions d'intégration

Le workflow proposé effectue une installation de dépendances CPU et une matrice Ubuntu/Windows × Python 3.12/3.13. Il conserve les JUnit et les rapports JSON, dont les identifiants GitHub et les empreintes du code, des tests et des configurations. Une présence du workflow ou un badge ne prouve pas qu'une étape a été exécutée. Les résultats hébergés sont rattachés à la tête de PR et au checkout de fusion effectif dans la discussion de la PR.

Cette qualification locale ne couvre ni une installation neuve des dépendances, ni Windows, ni Python 3.12, ni CUDA. Les preuves hébergées ultérieures doivent lever ces inconnues individuellement. Une réussite CPU ne devient pas une preuve GPU. Une répétabilité locale ne garantit pas l'identité numérique entre systèmes ou versions.

Aucune fusion, release ou modification du PC personnel n'est impliquée. La revue décrite est une auto-relecture de l'assistant implémenteur, pas une approbation indépendante. L'évaluation walk-forward, la provenance des poids et la validité prédictive sur des données réelles restent des chantiers distincts.

## Références

- Baseline : https://github.com/mehdisafer/LSTMvsGRUvsAttention_v2.0/tree/2630b916b114e128a722f42b9b609348f8f4111a
- MSELoss et réduction par éléments : https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss
- Renommage des en-têtes dupliqués : https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
- R2 sur cible constante : https://scikit-learn.org/1.8/modules/model_evaluation.html
- Les Actions sont épinglées par SHA depuis les références officielles relevées le 11 novembre 2026.
