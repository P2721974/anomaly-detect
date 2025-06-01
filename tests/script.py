# tests/script.py

import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/datasets/100k.csv")

# Check overall
print("Shape:", df.shape)
print("NaNs:", df.isna().sum().sum())
print("Infs:", np.isinf(df).sum().sum())

# Check per column
print("\nNaNs per column:\n", df.isna().sum())
print("\nInfs per column:\n", np.isinf(df).sum())

# Optionally drop + save
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df.to_csv("data/processed/datasets/100k_clean.csv", index=False)