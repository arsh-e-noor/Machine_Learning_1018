import pandas as pd

mapping = {"Low": 1, "Medium": 2, "High": 3}
features = pd.DataFrame({"Level": ["Low", "Medium", "High", "Medium", "Low"]})

features["Level_Encoded"] = features["Level"].map(mapping)

print(features)
print(features["Level_Encoded"].value_counts())
