import pandas as pd

features = pd.DataFrame({"Region": ["North", "East", "South", "West", "North"]})
df_one_hot = pd.get_dummies(features, columns=["Region"], dtype=int)

print(df_one_hot)
