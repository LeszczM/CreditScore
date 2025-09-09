#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validate_models.py
Ładuje zapisane modele (pipeline: preproc + model) i porównuje je na *tym samym* holdoucie
zapisanym przez train_models.py (plik test_holdout.csv). Tworzy zestawienie metryk i wykresy.

Wejście:
  --models_dir    folder z modelami i artefaktami (domyślnie data/models)
  --target        nazwa kolumny celu (domyślnie 'default')

Wyjścia (do --models_dir):
  - model_comparison.csv            – zbiorcze metryki: AUC, Accuracy, Precision, Recall, F1, AP, Brier, KS
  - roc_<model>.png                 – krzywa ROC każdego modelu
  - pr_<model>.png                  – Precision-Recall każdego modelu
  - confusion_<model>.txt           – macierz pomyłek + classification_report
"""

import os, argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from joblib import load
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
    brier_score_loss, confusion_matrix, classification_report
)

def ks_threshold(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    ks = tpr - fpr
    i = int(np.argmax(ks))
    return float(thr[i]), float(ks[i]), float(fpr[i]), float(tpr[i])

def plot_roc(y_true, y_score, title, out_png):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0,1],[0,1], linestyle="--", label="Losowy")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(title); plt.legend(loc="lower right")
    plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()

def plot_pr(y_true, y_score, title, out_png):
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure()
    plt.plot(rec, prec, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(title); plt.legend(loc="lower left")
    plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", type=str, default="data/models")
    ap.add_argument("--target", type=str, default="default")
    args = ap.parse_args()

    test_path = os.path.join(args.models_dir, "test_holdout.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Brak {test_path}. Najpierw uruchom train_models.py")
    df_test = pd.read_csv(test_path)
    if args.target not in df_test.columns:
        raise ValueError(f"Brak kolumny celu '{args.target}' w {test_path}")

    y_test = df_test[args.target].astype(int)
    X_test = df_test.drop(columns=[args.target])

    # wczytaj modele
    model_files = sorted(glob(os.path.join(args.models_dir, "*.joblib")))
    if not model_files:
        raise FileNotFoundError(f"Nie znaleziono modeli w {args.models_dir}")

    rows = []
    for mf in model_files:
        name = os.path.splitext(os.path.basename(mf))[0]  # np. logit_en
        pipe = load(mf)
        p = pipe.predict_proba(X_test)[:, 1]

        # metryki probabilistyczne
        auc = roc_auc_score(y_test, p)
        ap = average_precision_score(y_test, p)
        brier = brier_score_loss(y_test, p)
        thr, ks, fpr_at, tpr_at = ks_threshold(y_test, p)

        # metryki klasyfikacyjne przy progu KS (dla porównania)
        y_hat = (p >= thr).astype(int)
        acc = accuracy_score(y_test, y_hat)
        prec = precision_score(y_test, y_hat, zero_division=0)
        rec = recall_score(y_test, y_hat, zero_division=0)
        f1 = f1_score(y_test, y_hat, zero_division=0)

        rows.append({
            "model": name,
            "AUC": float(auc),
            "AP": float(ap),
            "Brier": float(brier),
            "KS": float(ks),
            "Accuracy@KS": float(acc),
            "Precision@KS": float(prec),
            "Recall@KS": float(rec),
            "F1@KS": float(f1),
            "Thr_KS": float(thr),
            "FPR@KS": float(fpr_at),
            "TPR@KS": float(tpr_at)
        })

        # wykresy + confusion
        plot_roc(y_test, p, f"ROC – {name}", os.path.join(args.models_dir, f"roc_{name}.png"))
        plot_pr(y_test, p, f"PR – {name}", os.path.join(args.models_dir, f"pr_{name}.png"))

        cm = confusion_matrix(y_test, y_hat)
        rep = classification_report(y_test, y_hat, digits=3)
        with open(os.path.join(args.models_dir, f"confusion_{name}.txt"), "w") as f:
            f.write(f"Threshold (KS): {thr:.6f}\n")
            f.write("Confusion matrix [ [TN FP]\n                    [FN TP] ]\n")
            f.write(str(cm) + "\n\n" + rep)

        print(f"[{name}] AUC={auc:.4f} | F1@KS={f1:.3f} | KS={ks:.3f}")

    pd.DataFrame(rows).sort_values("AUC", ascending=False).to_csv(
        os.path.join(args.models_dir, "model_comparison.csv"), index=False
    )
    print(f"[OK] Porównanie zapisane: {os.path.join(args.models_dir, 'model_comparison.csv')}")

if __name__ == "__main__":
    main()
