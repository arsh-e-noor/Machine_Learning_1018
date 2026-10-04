import pandas as pd

df_json = pd.read_json("../Data sets/attendance.json")

print("JSON Data:")
print(df_json.head())
