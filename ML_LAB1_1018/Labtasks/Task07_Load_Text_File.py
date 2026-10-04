import pandas as pd

df_text = pd.read_csv("../Data sets/data.txt", delimiter=",")
print("Text File Data:")
print(df_text.head())
