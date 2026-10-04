import pandas as pd

json_url = "https://jsonplaceholder.typicode.com/users"
df_online_json = pd.read_json(json_url)

print(df_online_json.head())
