"""
univariate_auc.py
Liczy jednocechowe AUC dla wszystkich kolumn względem celu (0/1) i zapisuje wyniki + wykresy.
- Wejście:  CSV z danymi (domyślnie data/preprocessedData.csv)
- Cel:      kolumna binarna (domyślnie 'default')
- Wyjście:  data/univariate_auc.csv
            data/univariate_auc_topK.png
            data/auc_plots/<feature>_roc.png  (dla TOP-K cech wg AUC zorientowanego)
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_categorical_dtype

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------- Pomocnicze ----------

def is_categorical_series(s: pd.Series) -> bool:
    return s.dtype == "object" or is_categorical_dtype(s)

def target_mean_encode(s: pd.Series, y: pd.Series, alpha: float = 20.0) -> pd.Series:
    """
    Prosty target-mean encoding z wygładzeniem:
    score(cat) = (sum(y_cat) + prior*alpha) / (count_cat + alpha),
    gdzie prior = średni rate klasy pozytywnej w całej próbce.
    """
    prior = float(y.mean())
    df = pd.DataFrame({"x": s, "y": y}).dropna()
    if df.empty:
        return pd.Series(index=s.index, dtype="float64")

    stats = df.groupby("x")["y"].agg(["sum", "count"])
    # unikamy zer w mianowniku
    smoothed = (stats["sum"] + prior * alpha) / (stats["count"] + alpha)
    return s.map(smoothed)

def compute_univariate_auc(x: pd.Series, y: pd.Series, alpha_smooth: float = 20.0):
    """
    Zwraca tuple: (auc, auc_oriented, score_vector) dla pojedynczej kolumny x (po dopasowaniu do y).
    - liczby/bool: AUC na surowym x,
    - kategorie: AUC na score z target-mean encoding.
    """
    mask = (~x.isna()) & (~y.isna())
    x_ = x[mask]
    y_ = y[mask].astype(int)

    # wymagane dwie klasy i >= 10 obserwacji
    if y_.nunique() < 2 or len(y_) < 10:
        return (np.nan, np.nan, pd.Series(index=x.index, dtype="float64"))

    if is_numeric_dtype(x_) or is_bool_dtype(x_):
        score = pd.to_numeric(x_, errors="coerce")
    else:
        score = target_mean_encode(x_.astype("object"), y_, alpha=alpha_smooth)

    # jeśli score jest stały → AUC nieokreślone
    if score.nunique(dropna=True) < 2:
        return (np.nan, np.nan, score.reindex(x.index))

    auc = roc_auc_score(y_, score)
    auc_oriented = max(auc, 1 - auc)
    # zwróć score w oryginalnym indeksie (ułatwia rysowanie ROC per cecha)
    return (float(auc), float(auc_oriented), score.reindex(x.index))

def plot_roc(y_true: pd.Series, score: pd.Series, title: str, path_png: str):
    """Zapisuje wykres ROC dla danego score'u."""
    mask = (~y_true.isna()) & (~score.isna())
    y = y_true[mask].astype(int)
    s = score[mask]
    if y.nunique() < 2 or s.nunique() < 2:
        return  # nic nie rysujemy

    fpr, tpr, _ = roc_curve(y, s)
    auc = roc_auc_score(y, s)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Losowy")
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.title(title)
    plt.legend(loc="lower right")
    os.makedirs(os.path.dirname(path_png), exist_ok=True)
    plt.savefig(path_png, bbox_inches="tight", dpi=150)
    plt.close()

def barplot_topk(df_auc: pd.DataFrame, topk: int, out_png: str):
    """Wykres słupkowy TOP-K cech wg auc_oriented."""
    tmp = df_auc.dropna(subset=["auc_oriented"]).head(topk)
    if tmp.empty:
        return
    # odwróć kolejność, by najwyższe były na górze
    tmp = tmp.iloc[::-1]
    plt.figure(figsize=(8, max(4, 0.35 * len(tmp))))
    plt.barh(tmp["feature"], tmp["auc_oriented"])
    plt.xlabel("AUC (zorientowane)")
    plt.title(f"Jednocechowe AUC – TOP {len(tmp)}")
    for i, v in enumerate(tmp["auc_oriented"]):
        plt.text(v + 0.005, i, f"{v:.3f}", va="center")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close()

# ---------- Główna logika ----------

def main():
    parser = argparse.ArgumentParser(description="Jednocechowe AUC dla wszystkich kolumn względem celu (0/1).")
    parser.add_argument("--input", type=str, default="data/preprocessedData.csv", help="Ścieżka do CSV z danymi.")
    parser.add_argument("--target", type=str, default="default", help="Nazwa kolumny celu (0/1).")
    parser.add_argument("--outdir", type=str, default="data", help="Folder wyjściowy na wyniki/wykresy.")
    parser.add_argument("--alpha", type=float, default=20.0, help="Wygładzenie dla target-mean encoding (kategorie).")
    parser.add_argument("--topk", type=int, default=20, help="Ile ROC-ów zapisać (TOP-K wg auc_oriented). -1 = wszystkie.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Wczytaj dane
    df = pd.read_csv(args.input,index_col=0)
    if args.target not in df.columns:
        raise ValueError(f"Nie znaleziono kolumny celu '{args.target}' w {args.input}.")
    y_all = df[args.target]
    # binarny cel 0/1
    try:
        y_all = pd.to_numeric(y_all, errors="raise").astype(int)
    except Exception:
        raise ValueError(f"Kolumna celu '{args.target}' musi być liczbowa 0/1.")

    if set(pd.Series(y_all.dropna().unique())) - {0, 1}:
        raise ValueError(f"Kolumna celu '{args.target}' musi być binarna (0/1).")

    feature_cols = [c for c in df.columns if c != args.target]

    # 2) Policz AUC per cecha
    rows = []
    per_feature_scores = {}  # do późniejszego rysowania ROC
    for col in feature_cols:
        auc, auc_or, score = compute_univariate_auc(df[col], y_all, alpha_smooth=args.alpha)
        rows.append({
            "feature": col,
            "dtype": str(df[col].dtype),
            "n": int(len(df[col]) - df[col].isna().sum()),
            "n_pos": int((y_all == 1).sum()),
            "n_neg": int((y_all == 0).sum()),
            "auc": auc if np.isfinite(auc) else np.nan,
            "auc_oriented": auc_or if np.isfinite(auc_or) else np.nan,
            "note": "" if np.isfinite(auc_or) else "za mało klas/obserwacji lub stały score"
        })
        per_feature_scores[col] = score

    df_auc = pd.DataFrame(rows).sort_values("auc_oriented", ascending=False, na_position="last").reset_index(drop=True)

    # 3) Zapis tabeli
    out_csv = os.path.join(args.outdir, "univariate_auc.csv")
    df_auc.to_csv(out_csv, index=False)
    print(f"[OK] Zapisano tabelę AUC -> {out_csv}")

    # 4) Wykres słupkowy TOP-K
    topk = args.topk
    if topk == -1:
        topk = df_auc["auc_oriented"].notna().sum()
    top_png = os.path.join(args.outdir, "univariate_auc_topK.png")
    barplot_topk(df_auc, topk, top_png)
    print(f"[OK] Zapisano wykres TOP-K -> {top_png}")

    # 5) ROC dla TOP-K cech
    roc_dir = os.path.join(args.outdir, "auc_plots")
    os.makedirs(roc_dir, exist_ok=True)
    top_features = df_auc.dropna(subset=["auc_oriented"]).head(topk)["feature"].tolist()
    for feat in top_features:
        score = per_feature_scores.get(feat, pd.Series(dtype="float64"))
        png_path = os.path.join(roc_dir, f"{feat}_roc.png")
        title = f"ROC – {feat}"
        plot_roc(y_all, score, title, png_path)
    print(f"[OK] Zapisano ROC-e dla {len(top_features)} cech -> {roc_dir}")

    # 6) Dodatkowo: szybkie podsumowanie do konsoli
    print(df_auc.head(min(10, len(df_auc))).to_string(index=False))

if __name__ == "__main__":
    main()