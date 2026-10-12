# ============================================================================
#  Name             : ARSHENOOR
#  Registration No. : 23-NTU-CS-1018
#  Section          : BSCS 7th A
#  Course           : Machine Learning Lab
#  Lab              : Lab 04  -  SVM (Support Vector Machine)
#  Dataset          : income_evaluation.csv
# ============================================================================

# Import all relevant libraries
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Load the dataset
# ---------------------------------------------------------------------------
df = pd.read_csv("income_evaluation.csv")

print(df.head())

# ---------------------------------------------------------------------------
# Identify categorical and numerical features
# ---------------------------------------------------------------------------
categorical_features = df.select_dtypes(include=['object']).columns
numerical_features = df.select_dtypes(exclude=['object']).columns

# Apply one-hot encoding to categorical features
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Separate features and target variable again after one-hot encoding
X = df.drop("income_ >50K", axis=1)   # Adjust target column name based on get_dummies output
y = df["income_ >50K"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101)

# Scale numerical features
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

print(X_train.head())

# ---------------------------------------------------------------------------
# Train the SVM
# ---------------------------------------------------------------------------
svc = SVC(random_state=101)
svc.fit(X_train, y_train)

accuracies = cross_val_score(svc, X_train, y_train, cv=5)

print("Train Score:", np.mean(accuracies))
print("Test Score:", svc.score(X_test, y_test))

# ---------------------------------------------------------------------------
# Tuning with GridSearchCV
# ---------------------------------------------------------------------------
from sklearn.model_selection import GridSearchCV

grid = {
    'C': [0.01, 0.1, 1, 10],
    'kernel': ["linear", "poly", "rbf", "sigmoid"],
    'degree': [1, 3, 5, 7],
    'gamma': [0.01, 1]
}

# --- smaller, practical version: comment the grid above and use this one ---
# grid = {
#     'C': [1, 10],
#     'kernel': ["rbf"],
#     'gamma': [0.01, 1]
# }

svm = SVC()
svm_cv = GridSearchCV(svm, grid, cv=5)
svm_cv.fit(X_train, y_train)

print("Best Parameters:", svm_cv.best_params_)
print("Train Score:", svm_cv.best_score_)
print("Test Score:", svm_cv.score(X_test, y_test))