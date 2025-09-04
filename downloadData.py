import pandas as pd
from sklearn.datasets import fetch_openml

# 1) Pobierz dane z OpenML
ds = fetch_openml(name="credit-g", version=1, as_frame=True)
df = ds.frame.copy()

# 2) Ujednolicenie celu: class -> default (1=bad, 0=good)
if "class" in df.columns:
    df["default"] = (df["class"].str.lower() == "bad").astype(int)
    df.drop(columns=["class"], inplace=True)

# 3) Zapis do CSV (ścieżka zgodna z wcześniejszym raportem)
out_path = "data/german_credit_data.csv"
df.to_csv(out_path, index=False)
print(f"Saved to {out_path}, shape={df.shape}")