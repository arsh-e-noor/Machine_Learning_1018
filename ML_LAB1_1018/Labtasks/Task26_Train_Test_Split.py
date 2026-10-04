from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

np.random.seed(42)
X = np.random.rand(1000, 2)
y = pd.Series([0] * 900 + [1] * 100)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)
