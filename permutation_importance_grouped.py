#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
permutation_importance_grouped.py
Permutation Importance (spadek AUC) z grupowaniem po oryginalnych polach,
działające na PIPELINE (pre + OHE + model). Permutujemy kolumny oryginalnego DataFrame,
dzięki czemu automatycznie grupujemy wszystkie dummye OHE w jedną cechę.

Wejście:
  --input   : CSV z danymi (np. data/preprocessedData.csv)
  --target  : nazwa kolumny celu (0/1), np. 'default'
  --model   : ścieżka do zapisanego pipeline'u joblib (np. data/best_model.joblib)
  --outdir  : katalog wyjściowy na wyniki/wykresy (np. data/perm_imp)

Wyjście:
  outdir/permutation_importance.csv   – tabela: feature, mean_drop_auc, std_drop_auc, base_auc
  outdir/permutation_importance.png   – barh TOP-K spadków AUC

Uwaga:
  - Liczymy na zbiorze TEST (holdout) – nie na treningu.
  - Dla każdej cechy powtarzamy permutację n_repeats razy i uśredniamy spadek AUC.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def barh_plot(names, values, title, out_png, topk=30):
    order = np.argsort(values)[::-1][:topk]
    names_top = [names[i] for i in order][::-1]
    vals_top  = [values[i] for i in order][::-1]
    plt.figure(figsize=(9, max(4, 0.35*len(names_top))))
    plt.barh(names_top, vals_top)
    for i, v in enumerate(vals_top):
        plt.text(v + 0.002, i, f"{v:.3f}", va="center")
    plt.xlabel("Spadek AUC (base_auc − auc_po_permutacji)")
    plt.title(title)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="data/preprocessedData.csv")
    ap.add_argument("--target", type=str, default="default")
    ap.add_argument("--model", type=str, required=True, help="Ścieżka do joblib z pipeline'em (pre + clf).")
    ap.add_argument("--outdir", type=str, default="data/perm_imp")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--n_repeats", type=int, default=10, help="Powtórzenia permutacji na cechę.")
    ap.add_argument("--topk", type=int, default=30, help="Ile cech pokazać na wykresie.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.random_state)

    # 1) Dane
    df = pd.read_csv(args.input)
    if args.target not in df.columns:
        raise ValueError(f"Brak kolumny celu '{args.target}' w {args.input}")

    y_all = pd.to_numeric(df[args.target], errors="coerce")
    if set(y_all.dropna().unique()) - {0, 1}:
        raise ValueError(f"Kolumna celu '{args.target}' musi być binarna (0/1).")

    X_all = df.drop(columns=[args.target])
    # wyrzuć wiersze z NaN w y
    mask = ~y_all.isna()
    X_all = X_all.loc[mask].reset_index(drop=True)
    y_all = y_all.loc[mask].astype(int).reset_index(drop=True)

    # 2) Wczytaj PIPELINE (pre + clf)
    pipe = load(args.model)

    # 3) Split – TEST do permutacji
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=args.test_size, stratify=y_all, random_state=args.random_state
    )

    # (opcjonalnie) dopasowanie, gdyby pipeline był nieprzetrenowany – zakładamy, że jest już wytrenowany
    # pipe.fit(X_train, y_train)

    # 4) AUC bazowe na teście
    p_base = pipe.predict_proba(X_test)[:, 1]
    base_auc = roc_auc_score(y_test, p_base)

    # 5) Lista cech do permutacji (oryginalne kolumny DF – to jest nasze "grupowanie")
    features = X_test.columns.tolist()

    drops = []
    drops_std = []
    for feat in features:
        aucs = []
        for r in range(args.n_repeats):
            Xp = X_test.copy()
            # permutacja TYLKO jednej kolumny (oryginalnej cechy)
            Xp[feat] = rng.permutation(Xp[feat].values)
            # predykcja przez cały pipeline (pre + OHE + model)
            p = pipe.predict_proba(Xp)[:, 1]
            aucs.append(roc_auc_score(y_test, p))
        aucs = np.array(aucs, dtype=float)
        drops.append(base_auc - aucs.mean())
        drops_std.append(aucs.std(ddof=1))

    # 6) Tabela wyników
    df_imp = pd.DataFrame({
        "feature": features,
        "mean_drop_auc": drops,
        "std_auc": drops_std,
        "base_auc": base_auc
    }).sort_values("mean_drop_auc", ascending=False).reset_index(drop=True)

    out_csv = os.path.join(args.outdir, "permutation_importance.csv")
    df_imp.to_csv(out_csv, index=False)

    # 7) Wykres TOP-K
    out_png = os.path.join(args.outdir, "permutation_importance.png")
    barh_plot(df_imp["feature"].tolist(), df_imp["mean_drop_auc"].to_numpy(),
              title="Permutation Importance (grupowanie po oryginalnych polach)",
              out_png=out_png, topk=args.topk)

    print(f"[OK] base AUC (test): {base_auc:.4f}")
    print(f"[OK] Wyniki zapisane: {out_csv}")
    print(f"[OK] Wykres zapisany: {out_png}")

if __name__ == "__main__":
    main()
