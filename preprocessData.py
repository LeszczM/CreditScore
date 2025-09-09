import pandas as pd
import numpy as np

PATH = "data/german_credit_data.csv"

# 1) Wczytaj
df_raw = pd.read_csv(PATH)
f = open("preprocessReport",'w+',encoding="utf-8")

print("=== Podstawowe info o dataframe ===",file = f)
print(f"Plik: {PATH}",file = f)
print(f"Liczba wierszy x kolumn:", df_raw.shape,file = f)
print(f"Kolumny:", df_raw.columns.tolist(),file = f)

# 2) Znajdź/utwórz kolumnę celu: 'default' (1 = bad/default, 0 = good)
df = df_raw.copy()

target_col = None
for cand in ["default", "class", "target", "y"]:
    if cand in df.columns:
        target_col = cand
        break

if target_col is None:
    raise ValueError(
        "Nie znaleziono kolumny celu. Oczekuję 'default' (0/1) lub 'class' (good/bad). "
        f"Dostępne kolumny: {df.columns.tolist()}"
    )

# Ujednolicenie celu -> 'default' w 0/1
if target_col == "class":
    # najczęstszy wariant: 'good' / 'bad'
    vals = df["class"].astype(str).str.lower()
    if {"good", "bad"}.issubset(set(vals.unique())):
        df["default"] = (vals == "bad").astype(int)
    else:
        # jeśli to nie good/bad, spróbuj rzutować na 0/1
        df["default"] = pd.to_numeric(df["class"], errors="coerce").fillna(0).astype(int)
    df.drop(columns=["class"], inplace=True)
elif target_col != "default":
    # np. 'target' -> przenieś do 'default'
    df["default"] = pd.to_numeric(df[target_col], errors="coerce")
    df.drop(columns=[target_col], inplace=True)

# 3) Usuń brakujące w celu i upewnij się, że binarny
df = df.dropna(subset=["default"]).reset_index(drop=True)
df["default"] = df["default"].astype(int)

uniq = sorted(df["default"].unique().tolist())
if not set(uniq).issubset({0, 1}):
    raise ValueError(f"Kolumna 'default' nie jest binarna. Unikalne wartości: {uniq}")

# 4) Szybki przegląd danych
print("\n=== Typy danych ===",file = f)
print(df.dtypes,file = f)

print("\n=== Braki danych ===",file = f)
na_counts = df.isna().sum().sort_values(ascending=False)
print(na_counts[na_counts > 0],file = f)

print("\n=== Balans klas (0=good, 1=bad/default) ===",file = f)
class_counts = df["default"].value_counts(dropna=False).rename({0: "good", 1: "bad"})
print(class_counts,file = f)
rate_bad = df["default"].mean()
print(f"Default rate: {rate_bad:.3%}",file = f)

# 5) Identyfikacja typów cech (przyda się później do preprocessingu)
feature_cols = [c for c in df.columns if c != "default"]
cat_cols = [c for c in feature_cols if df[c].dtype == "object"]
num_cols = [c for c in feature_cols if df[c].dtype != "object"]

print("=== Podział kolumn ===",file = f)
print("Numeryczne:", num_cols,file = f)
print("Kategoryczne:", cat_cols,file = f)

f.close()

df.to_csv('data/preprocessedData.csv', index=False)