#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_models.py  (wersja z XGBoost)
Trenuje kilka klasycznych modeli ML na TOP-N cechach wskazanych w pliku CSV z rankingiem.
Cechy wybierane są po nazwach **oryginalnych kolumn** (przed OHE).

Wejście:
  --input         CSV z danymi (X + target), domyślnie data/preprocessedData.csv
  --target        nazwa kolumny celu (0/1), domyślnie 'default'
  --features_csv  CSV z rankingiem cech (kol. 'feature' lub 'orig_feature')
  --topn          ile najlepszych cech wziąć, domyślnie 10

Wyjścia (do --outdir, domyślnie data/models/):
  - models/*.joblib                – najlepsze modele (pipeline: preproc + model)
  - cv_results_<model>.csv
  - best_params_<model>.json
  - train_cv_summary.csv
  - selected_features.json
  - test_holdout.csv               – ten sam holdout dla wszystkich modeli
  - split_info.json
"""

import os, json, argparse
import numpy as np
import pandas as pd

from joblib import dump
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

# NEW: XGBoost
from xgboost import XGBClassifier

def read_top_features(path_csv: str, topn: int) -> list[str]:
    df = pd.read_csv(path_csv)
    col = None
    for cand in ["feature", "orig_feature"]:
        if cand in df.columns:
            col = cand
            break
    if col is None:
        raise ValueError("Plik z rankingiem musi mieć kolumnę 'feature' lub 'orig_feature'.")
    feats = df[col].dropna().astype(str).tolist()
    return feats[:topn]

def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False))])
    transformers = []
    if num_cols: transformers.append(("num", num_pipe, num_cols))
    if cat_cols: transformers.append(("cat", cat_pipe, cat_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)

def model_registry(random_state: int, scale_pos_weight: float):
    """Zwraca słownik: nazwa -> (estymator, param_grid)"""
    models = {
        "logit_en": (
            LogisticRegression(penalty="elasticnet", solver="saga",
                               class_weight="balanced", max_iter=10000, n_jobs=-1),
            {"clf__l1_ratio": [0.2, 0.5, 0.8],
             "clf__C": [0.05, 0.1, 0.2, 0.5, 1.0]}
        ),
        "rf": (
            RandomForestClassifier(n_estimators=400, random_state=random_state,
                                   class_weight="balanced_subsample", n_jobs=-1),
            {"clf__max_depth": [None, 6, 10, 16],
             "clf__min_samples_leaf": [1, 2, 5],
             "clf__max_features": ["sqrt", 0.5, 0.8]}
        ),
        "hgb": (
            HistGradientBoostingClassifier(random_state=random_state,
                                           learning_rate=0.1,
                                           max_depth=None,
                                           class_weight="balanced"),
            {"clf__learning_rate": [0.05, 0.1, 0.2],
             "clf__max_leaf_nodes": [31, 63, 127],
             "clf__max_depth": [None, 6, 10]}
        ),
        "svc": (
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state),
            {"clf__C": [0.5, 1.0, 2.0],
             "clf__gamma": ["scale", 0.1, 0.01]}
        ),
        # NEW: XGBoost (drzewiasty, obsługuje nierównowagę przez scale_pos_weight)
        "xgb": (
            XGBClassifier(
                random_state=random_state,
                n_estimators=400,
                learning_rate=0.1,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=1.0,
                reg_lambda=1.0,
                reg_alpha=0.0,
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight
            ),
            {
                "clf__n_estimators": [300, 400, 600],
                "clf__max_depth": [3, 4, 6],
                "clf__learning_rate": [0.05, 0.1],
                "clf__subsample": [0.7, 0.9],
                "clf__colsample_bytree": [0.7, 0.9],
                "clf__min_child_weight": [1.0, 3.0],
                "clf__reg_lambda": [1.0, 3.0],
                "clf__reg_alpha": [0.0, 0.5]
            }
        ),
    }
    return models

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="data/preprocessedData.csv")
    ap.add_argument("--target", type=str, default="default")
    ap.add_argument("--features_csv", type=str, required=True,
                    help="CSV z rankingiem cech (kolumna 'feature' lub 'orig_feature').")
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--outdir", type=str, default="data/models")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--cv_splits", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Dane i wybór cech
    df = pd.read_csv(args.input)
    if args.target not in df.columns:
        raise ValueError(f"Brak kolumny celu '{args.target}' w {args.input}")
    y_all = pd.to_numeric(df[args.target], errors="coerce")
    if set(y_all.dropna().unique()) - {0, 1}:
        raise ValueError(f"Kolumna celu '{args.target}' musi być 0/1.")
    X_all = df.drop(columns=[args.target])

    feats_top = read_top_features(args.features_csv, args.topn)
    missing = [f for f in feats_top if f not in X_all.columns]
    if missing:
        raise ValueError(f"Następujących cech nie ma w danych: {missing}")
    X_all = X_all[feats_top]

    mask = ~y_all.isna()
    X_all = X_all.loc[mask].reset_index(drop=True)
    y_all = y_all.loc[mask].astype(int).reset_index(drop=True)

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=args.test_size, stratify=y_all, random_state=args.random_state
    )

    # 3) Preprocessor
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    pre = build_preprocessor(num_cols, cat_cols)

    # 4) Klasyfikatory + CV (oblicz scale_pos_weight dla XGB)
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    spw = float(neg) / float(pos) if pos > 0 else 1.0

    models = model_registry(args.random_state, spw)
    cv = StratifiedKFold(n_splits=args.cv_splits, shuffle=True, random_state=args.random_state)
    rows = []

    for name, (clf, grid) in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc",
                          n_jobs=-1, refit=True, verbose=0, return_train_score=False)
        gs.fit(X_train, y_train)

        # zapis
        model_path = os.path.join(args.outdir, f"{name}.joblib")
        dump(gs.best_estimator_, model_path)
        pd.DataFrame(gs.cv_results_).sort_values("rank_test_score").to_csv(
            os.path.join(args.outdir, f"cv_results_{name}.csv"), index=False)
        with open(os.path.join(args.outdir, f"best_params_{name}.json"), "w") as f:
            json.dump(gs.best_params_, f, indent=2)

        best_cv_auc = float(np.max(gs.cv_results_["mean_test_score"]))
        p_test = gs.best_estimator_.predict_proba(X_test)[:, 1]
        auc_holdout = roc_auc_score(y_test, p_test)

        rows.append({"model": name, "cv_auc_mean": best_cv_auc, "holdout_auc": float(auc_holdout)})
        print(f"[{name}] CV AUC={best_cv_auc:.4f} | Holdout AUC={auc_holdout:.4f} -> {model_path}")

    # 5) Zbiorcze podsumowanie
    pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False).to_csv(
        os.path.join(args.outdir, "train_cv_summary.csv"), index=False
    )

    # 6) Artefakty do walidacji
    with open(os.path.join(args.outdir, "selected_features.json"), "w") as f:
        json.dump({"features": feats_top}, f, indent=2)
    test_out = X_test.copy()
    test_out[args.target] = y_test.values
    test_out.to_csv(os.path.join(args.outdir, "test_holdout.csv"), index=False)
    with open(os.path.join(args.outdir, "split_info.json"), "w") as f:
        json.dump({"test_size": args.test_size, "random_state": args.random_state}, f, indent=2)

    print(f"[OK] Artefakty zapisane w: {args.outdir}")

if __name__ == "__main__":
    main()
