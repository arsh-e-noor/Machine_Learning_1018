import pandas as pd
import numpy as np

np.random.seed(42)
y = pd.Series([0] * 900 + [1] * 100)

print(y.value_counts())
