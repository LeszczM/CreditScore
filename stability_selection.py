#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stability_selection.py
Stability selection dla LogisticRegression (SAGA) z karą L1/Elastic Net.
- Wczytuje data/preprocessedData.csv (domyślnie), cel binarny (0/1).
- Prefituje preprocessor (imputacja + scaler + OHE), transformuje X do stałej macierzy cech.
- B-krotnie losuje podpróbkę (stratyfikowaną), trenuje logit i zlicza, które wagi są ≠ 0.
- Zapisuje:
    outdir/stability_transformed.csv       (częstotliwość per kolumna po transformacji)
    outdir/stability_grouped.csv           (częstotliwość po oryginalnych polach)
    outdir/topK_grouped.png                (wykres TOP-K po częstotliwości, grupy)
    outdir/topK_transformed.png            (wykres TOP-K po częstotliwości, kolumny po transformacji)
Uwaga: to selekcja cech, nie ocena metryk. Preprocessing prefituje się globalnie, żeby kolumny OHE były spójne w każdej iteracji.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import check_random_state
from sklearn.metrics import roc_auc_score

# ---------- pomocnicze ----------

def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))
    ])
    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    pre = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)
    return pre

def robust_group_map(feature_names, num_cols, cat_cols):
    """
    Zwraca listę 'orig_name_by_index' (o długości = liczba kolumn po transformacji),
    która mapuje każdą kolumnę po transformacji na ORYGINALNE pole (num albo nazwa kategorycznej).
    Robimy to po nazwach z get_feature_names_out(): 'num__col', 'cat__col_category'.
    """
    orig = []
    # szybki lookup dla kat kolumn, by nie parsować po '_' (które mogą być w nazwach kategorii)
    for name in feature_names:
        if name.startswith("num__"):
            orig.append(name.split("__", 1)[1])
        elif name.startswith("cat__"):
            base = name.split("__", 1)[1]  # 'col_category'
            # znajdź takie col, które jest prefixem base + '_'
            found = None
            for c in cat_cols:
                prefix = f"{c}_"
                if base.startswith(prefix) or base == c:
                    found = c
                    break
            orig.append(found if found is not None else base.split("_", 1)[0])
        else:
            # fallback
            orig.append(name)
    return orig

def barh_plot(names, values, title, out_png, topk=30):
    idx = np.argsort(values)[::-1][:topk]
    names_top = [names[i] for i in idx]
    vals_top = [values[i] for i in idx]
    # odwróć kolejność, by największe były u góry
    names_top = names_top[::-1]
    vals_top = vals_top[::-1]
    plt.figure(figsize=(9, max(4, 0.35*len(names_top))))
    plt.barh(names_top, vals_top)
    for i, v in enumerate(vals_top):
        plt.text(v + 0.005, i, f"{v:.2f}", va="center")
    plt.xlabel("Częstotliwość wyboru (0–1)")
    plt.title(title)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# ---------- główna funkcja ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="data/preprocessedData.csv", help="Ścieżka do CSV.")
    ap.add_argument("--target", type=str, default="default", help="Kolumna celu (0/1).")
    ap.add_argument("--outdir", type=str, default="data/stability", help="Folder wyjściowy.")
    ap.add_argument("--iters", type=int, default=50, help="Liczba iteracji stability selection (B).")
    ap.add_argument("--subsample", type=float, default=0.75, help="Udział próby w każdej iteracji (0–1).")
    ap.add_argument("--penalty", type=str, default="l1", choices=["l1", "elasticnet"], help="Rodzaj kary.")
    ap.add_argument("--l1_ratio", type=float, default=1.0, help="Miks L1/L2 dla elasticnet (0–1). Ignorowane przy l1.")
    ap.add_argument("--C", type=float, default=0.3, help="Siła regularyzacji (mniejszy C => mocniejsza kara).")
    ap.add_argument("--random_state", type=int, default=42, help="Seed.")
    ap.add_argument("--class_weight", type=str, default="balanced", help="class_weight dla LogisticRegression.")
    ap.add_argument("--topk", type=int, default=30, help="Ile pozycji na wykresach.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = check_random_state(args.random_state)

    # 1) Wczytanie
    df = pd.read_csv(args.input)
    if args.target not in df.columns:
        raise ValueError(f"Brak kolumny celu '{args.target}' w {args.input}")
    y = pd.to_numeric(df[args.target], errors="coerce")
    if set(y.dropna().unique()) - {0, 1}:
        raise ValueError(f"Kolumna '{args.target}' musi być binarna (0/1).")
    feat_cols = [c for c in df.columns if c != args.target]
    X_df = df[feat_cols]
    # typy
    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # usuń NaN w y
    mask = ~y.isna()
    y = y.loc[mask].astype(int).reset_index(drop=True)
    X_df = X_df.loc[mask, :].reset_index(drop=True)

    # 2) Prefit preprocessora i transformacja X -> stała macierz
    pre = build_preprocessor(num_cols, cat_cols)
    pre.fit(X_df)  # UWAGA: prefit na całości wyłącznie do STABILITY (spójność kolumn po OHE)
    X_all = pre.transform(X_df)  # macierz (zwykle rzadka)
    feature_names = pre.get_feature_names_out().tolist()
    n_features = X_all.shape[1]

    # mapowanie kolumn po transformacji do oryginalnych pól
    orig_by_index = robust_group_map(feature_names, num_cols, cat_cols)

    # 3) Pętla stability selection (B iteracji)
    select_counts = np.zeros(n_features, dtype=float)
    sum_abs_betas = np.zeros(n_features, dtype=float)

    splitter = StratifiedShuffleSplit(n_splits=args.iters, train_size=args.subsample,
                                      random_state=args.random_state)

    # przygotuj classifier
    penalty = args.penalty
    l1_ratio = args.l1_ratio if penalty == "elasticnet" else None

    # Iteracje
    for b, (idx, _) in enumerate(splitter.split(np.zeros(len(y)), y), start=1):
        Xb = X_all[idx]
        yb = y.iloc[idx].values

        clf = LogisticRegression(
            penalty=penalty,
            solver="saga",
            l1_ratio=l1_ratio,
            C=args.C,
            max_iter=10000,
            class_weight=args.class_weight,
            n_jobs=-1
        )
        clf.fit(Xb, yb)

        beta = clf.coef_.ravel()
        mask_nz = np.abs(beta) > 1e-12
        select_counts[mask_nz] += 1.0
        sum_abs_betas += np.abs(beta)

        # (opcjonalnie) szybki log z uogólnieniem siły
        # if b % 10 == 0:
        #     print(f"[{b}/{args.iters}] wybrane: {mask_nz.sum()} kolumn")

    # 4) Agregacje i zapisy
    freq = select_counts / float(args.iters)
    mean_abs_beta = sum_abs_betas / float(args.iters)

    df_trans = pd.DataFrame({
        "feature_transformed": feature_names,
        "orig_feature": orig_by_index,
        "freq_selected": freq,
        "mean_abs_beta": mean_abs_beta
    }).sort_values(["freq_selected", "mean_abs_beta"], ascending=False)

    df_trans.to_csv(os.path.join(args.outdir, "stability_transformed.csv"), index=False)

    # Grupowanie po oryginalnych polach (sum/any po dummach)
    grouped = df_trans.groupby("orig_feature").agg(
        freq_any=("freq_selected", "mean"),   # średnia po kolumnach może zaniżać; alternatywnie: odsetek iteracji z "jakimkolwiek" ≠ 0
        freq_max=("freq_selected", "max"),
        k_selected=("freq_selected", lambda s: (s > 0).sum()),
        k_total=("freq_selected", "size"),
        mean_abs_beta_sum=("mean_abs_beta", "sum"),
        mean_abs_beta_max=("mean_abs_beta", "max")
    ).reset_index()

    # Dla czytelności przyjmijmy „częstotliwość grupy” = max po jej kolumnach (czy jakikolwiek dummy był wybierany często)
    grouped = grouped.rename(columns={"freq_max": "group_freq"})
    grouped = grouped.sort_values("group_freq", ascending=False)
    grouped.to_csv(os.path.join(args.outdir, "stability_grouped.csv"), index=False)

    # 5) Wykresy TOP-K
    barh_plot(
        names=grouped["orig_feature"].tolist(),
        values=grouped["group_freq"].to_numpy(),
        title="Stability selection – TOP cechy (po oryginalnych polach)",
        out_png=os.path.join(args.outdir, "topK_grouped.png"),
        topk=args.topk
    )
    barh_plot(
        names=df_trans["feature_transformed"].tolist(),
        values=df_trans["freq_selected"].to_numpy(),
        title="Stability selection – TOP kolumny (po transformacji)",
        out_png=os.path.join(args.outdir, "topK_transformed.png"),
        topk=args.topk
    )

    # 6) Krótki raport JSON z parametrami
    with open(os.path.join(args.outdir, "run_params.json"), "w") as f:
        json.dump({
            "iters": args.iters,
            "subsample": args.subsample,
            "penalty": args.penalty,
            "l1_ratio": args.l1_ratio if args.penalty == "elasticnet" else None,
            "C": args.C,
            "random_state": args.random_state,
            "class_weight": args.class_weight,
            "n_features_transformed": int(n_features),
            "n_num": len(num_cols),
            "n_cat": len(cat_cols)
        }, f, indent=2)

    print(f"[OK] Zapisano:\n - stability_transformed.csv\n - stability_grouped.csv\n - topK_grouped.png\n - topK_transformed.png\n(do folderu: {args.outdir})")

if __name__ == "__main__":
    main()
