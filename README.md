# FinGuard AI — Détection de fraude aux paiements

Plateforme de détection de fraude : modèle ML, scoring de risque explicable, API FastAPI, console analyste (Streamlit) et reporting Power BI.

## Démarrage
1. Télécharger le dataset Sparkov (Kaggle : *Credit Card Transactions Fraud Detection Dataset*) et placer `fraudTrain.csv` et `fraudTest.csv` dans `data/raw/`.
2. `make install` puis `make test` puis `make train`.

## Choix méthodologiques
- Split **temporel** 70/15/15 (jamais aléatoire), seuil choisi sur validation, évalué sur test.
- Features comportementales calculées avec le **passé strict** (testé dans `tests/test_features.py`).
- Déséquilibre géré par `class_weight` / `scale_pos_weight` (pas de SMOTE).
- Métriques : PR-AUC, rappel, précision et **coût attendu** (fraude manquée = montant perdu, fausse alerte = coût fixe).

## Limites
Données synthétiques : les performances ne se transposent pas telles quelles à une vraie banque.

## Feuille de route
- [x] Phase 1 : features, split temporel, baseline
- [ ] Phase 2 : SHAP + calibration + niveaux de risque
- [ ] Phase 3 : API FastAPI + Docker + tests
- [ ] Phase 4 : Cloud Run
- [ ] Phase 5 : console Streamlit
- [ ] Phase 6 : export Parquet + Power BI
- [ ] Phase 7 : MLflow, Evidently, GitHub Actions
