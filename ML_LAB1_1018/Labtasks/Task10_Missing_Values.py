import pandas as pd
import numpy as np

data = {
    "A": [1, 2, None, 4, 5],
    "B": [6, None, 8, 9, 10],
    "C": [11, 12, 13, None, 15]
}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)
print("\nMissing Values:")
print(df.isnull().sum())

df_dropna_rows = df.dropna(axis=0)
df_dropna_columns = df.dropna(axis=1)
df_fill_median = df.fillna(df.median(numeric_only=True))

print("\nAfter Dropping Rows:")
print(df_dropna_rows)
print("\nAfter Dropping Columns:")
print(df_dropna_columns)
print("\nAfter Filling with Median:")
print(df_fill_median)
