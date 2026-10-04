from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import pandas as pd

np.random.seed(42)
X = np.random.rand(1000, 2)
y = pd.Series([0] * 900 + [1] * 100)

sampler = RandomUnderSampler(random_state=42)
X_under, y_under = sampler.fit_resample(X, y)

print(pd.Series(y_under).value_counts())
