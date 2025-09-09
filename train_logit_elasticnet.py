#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_logit_elasticnet_ohe.py
Regresja logistyczna z Elastic Net + OneHotEncoder (dla kategorii).

Wejście:
  - CSV z danymi (domyślnie data/preprocessedData.csv), kolumna celu binarna (0/1), np. 'default'.
  - Skrypt sam wykrywa kolumny numeryczne i kategoryczne.
Wyjścia (do folderu --outdir, domyślnie data/):
  - best_model.joblib
  - best_params.json
  - cv_results.csv
  - test_metrics.json
  - roc_curve.png, pr_curve.png
  - coef_table.csv              (współczynniki po transformacji, z nazwami po OHE)
  - confusion_matrix_threshold_KS.txt
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score, precision_recall_curve,
    brier_score_loss, confusion_matrix, classification_report
)
from joblib import dump

# ---------- wykresy ----------
def plot_roc(y_true, y_score, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Losowy")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC (test)")
    plt.legend(loc="lower right")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_pr(y_true, y_score, out_path):
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure()
    plt.plot(rec, prec, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall (test)")
    plt.legend(loc="lower left")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def ks_threshold(y_true, y_score):
    """Zwróć próg maksymalizujący KS (= max(TPR-FPR)) i metryki w tym punkcie."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    ks_vals = tpr - fpr
    i = int(np.argmax(ks_vals))
    return float(thr[i]), {"KS": float(ks_vals[i]), "FPR_at_KS": float(fpr[i]), "TPR_at_KS": float(tpr[i])}

# ---------- główna logika ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="data/preprocessedData.csv", help="Ścieżka do CSV.")
    ap.add_argument("--target", type=str, default="default", help="Kolumna celu (0/1).")
    ap.add_argument("--test_size", type=float, default=0.2, help="Udział zbioru testowego.")
    ap.add_argument("--random_state", type=int, default=42, help="Seed losowy.")
    ap.add_argument("--outdir", type=str, default="data", help="Folder wyjściowy.")
    ap.add_argument("--cv_splits", type=int, default=5, help="Liczba foldów CV.")
    ap.add_argument("--l1_ratio", type=str, default="0.2,0.5,0.8", help="Lista l1_ratio, np. '0.2,0.5,0.8'.")
    ap.add_argument("--C_grid", type=str, default="0.05,0.1,0.2,0.5,1.0", help="Lista C, np. '0.05,0.1,0.2,0.5,1.0'.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Wczytaj dane
    df = pd.read_csv(args.input)
    if args.target not in df.columns:
        raise ValueError(f"Brak kolumny celu '{args.target}' w {args.input}")

    y = pd.to_numeric(df[args.target], errors="coerce")
    if set(y.dropna().unique()) - {0, 1}:
        raise ValueError(f"Kolumna celu '{args.target}' musi być binarna (0/1).")

    # Wykryj typy kolumn (X)
    # numeryczne = number; kategoryczne = object/category/bool (bool potraktujemy jako kategorię)
    feature_cols = [c for c in df.columns if c != args.target]
    dfX = df[feature_cols]
    num_cols = dfX.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = dfX.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if not num_cols and not cat_cols:
        raise ValueError("Nie znaleziono żadnych kolumn cech (num/cat).")

    # Usuń wiersze z NaN w y
    mask = ~y.isna()
    y = y.loc[mask].astype(int).reset_index(drop=True)
    X = dfX.loc[mask, :].reset_index(drop=True)

    # 2) Split train/test (stratyfikowany)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    # 3) Pipeline: imputacja + skalowanie (dla num) / imputacja + OHE (dla cat) + logit EN
    num_pipe = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())  # tylko dla num; OK bo num_pipe działa na macierzy gęstej
    ])

    cat_pipe = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))

    pre = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)

    pipe = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,        # nadpisane w GridSearch
            C=1.0,               # nadpisane w GridSearch
            max_iter=10000,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    # 4) CV: siatka po l1_ratio i C
    l1_list = [float(x) for x in args.l1_ratio.split(",") if x.strip()]
    C_list = [float(x) for x in args.C_grid.split(",") if x.strip()]
    param_grid = {"clf__l1_ratio": l1_list, "clf__C": C_list}

    cv = StratifiedKFold(n_splits=args.cv_splits, shuffle=True, random_state=args.random_state)
    grid = GridSearchCV(
        pipe, param_grid, cv=cv, scoring="roc_auc",
        n_jobs=-1, refit=True, verbose=0, return_train_score=False
    )
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    best_params = grid.best_params_
    cv_results = pd.DataFrame(grid.cv_results_).sort_values("rank_test_score")

    # 5) Ewaluacja na teście
    p_test = best.predict_proba(X_test)[:, 1]
    auc_test = roc_auc_score(y_test, p_test)
    brier = brier_score_loss(y_test, p_test)

    thr_ks, ks_info = ks_threshold(y_test, p_test)
    y_hat_ks = (p_test >= thr_ks).astype(int)
    cm = confusion_matrix(y_test, y_hat_ks)
    report = classification_report(y_test, y_hat_ks, digits=3)

    # 6) Zapisy artefaktów
    from joblib import dump
    dump(best, os.path.join(args.outdir, "best_model.joblib"))

    with open(os.path.join(args.outdir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)

    cv_results.to_csv(os.path.join(args.outdir, "cv_results.csv"), index=False)

    with open(os.path.join(args.outdir, "test_metrics.json"), "w") as f:
        json.dump({
            "AUC_test": auc_test,
            "Brier_test": brier,
            "threshold_KS": thr_ks,
            **ks_info
        }, f, indent=2)

    plot_roc(y_test, p_test, os.path.join(args.outdir, "roc_curve.png"))
    plot_pr(y_test, p_test, os.path.join(args.outdir, "pr_curve.png"))

    # 7) Tabela współczynników (po transformacjach – nazwy kolumn po OHE + num)
    pre_fit = best.named_steps["pre"]
    feat_names = []
    try:
        feat_names = pre_fit.get_feature_names_out().tolist()
    except Exception:
        # fallback: zbierz nazwy ręcznie
        feat_names = []
        if num_cols:
            feat_names += [f"num__{c}" for c in num_cols]
        if cat_cols:
            # dopasuj encoder i pobierz nazwy
            ohe = pre_fit.named_transformers_["cat"].named_steps["ohe"]
            cat_out = ohe.get_feature_names_out(cat_cols).tolist()
            feat_names += [f"cat__{n}" for n in cat_out]

    betas = best.named_steps["clf"].coef_.ravel()
    coef_table = pd.DataFrame({
        "feature_transformed": feat_names[:len(betas)],  # asekuracyjnie przy ewentualnych różnicach
        "beta": betas
    })
    coef_table["abs_beta"] = coef_table["beta"].abs()
    coef_table = coef_table.sort_values("abs_beta", ascending=False)
    coef_table.to_csv(os.path.join(args.outdir, "coef_table.csv"), index=False)

    with open(os.path.join(args.outdir, "confusion_matrix_threshold_KS.txt"), "w") as f:
        f.write(f"Threshold (KS): {thr_ks:.6f}\n")
        f.write("Confusion matrix [ [TN FP]\n                    [FN TP] ]\n")
        f.write(str(cm) + "\n\n")
        f.write(report)

    # 8) Log
    print(f"[OK] Best params: {best_params}")
    print(f"[OK] Test AUC: {auc_test:.4f} | Brier: {brier:.4f} | KS: {ks_info['KS']:.4f} @ thr={thr_ks:.4f}")
    print(f"[OK] Num cols: {len(num_cols)} | Cat cols: {len(cat_cols)}")
    print(f"[OK] Artefakty zapisane w: {args.outdir}")

if __name__ == "__main__":
    main()
