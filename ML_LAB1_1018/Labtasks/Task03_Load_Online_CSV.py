import pandas as pd

online_csv_url = "https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv"
df_online_csv = pd.read_csv(online_csv_url)

print(df_online_csv.head())
