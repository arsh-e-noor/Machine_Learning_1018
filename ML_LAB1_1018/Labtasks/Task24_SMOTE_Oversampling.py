from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd

np.random.seed(42)
X = np.random.rand(1000, 2)
y = pd.Series([0] * 900 + [1] * 100)

smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

print(pd.Series(y_smote).value_counts())
