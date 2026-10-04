import pandas as pd
from sklearn.preprocessing import LabelEncoder

features_df = pd.DataFrame({"Region": ["North", "East", "South", "West", "North"]})
le = LabelEncoder()

features_df["Region_Encoded"] = le.fit_transform(features_df["Region"])

print(features_df)
print(features_df["Region_Encoded"].value_counts())
