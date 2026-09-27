# ============================================================================
#  Name             : ARSHENOOR
#  Registration No. : 23-NTU-CS-1018
#  Section          : BSCS 7th A
#  Course           : Machine Learning Lab
#  Lab              : Lab 02  -  Classwork 04
#  Topic            : Logistic Regression using scikit-learn
#  Dataset          : Social_Network_Ads.csv
# ============================================================================

import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "Dataset", "Social_Network_Ads.csv")

# ---------------------------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------------------------
data = pd.read_csv(DATA_PATH)

print("First five rows of the dataset:")
print(data.head())

# Encode Gender (Male / Female -> 1 / 0)
le = LabelEncoder()
data["Gender"] = le.fit_transform(data["Gender"])

# ---------------------------------------------------------------------------
# 2. Features and target
# ---------------------------------------------------------------------------
X = data[["Gender", "Age", "EstimatedSalary"]].values
y = data["Purchased"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 3. Train the Logistic Regression model
# ---------------------------------------------------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

print("\nLearned coefficients :", model.coef_)
print("Learned intercept    :", model.intercept_)

# ---------------------------------------------------------------------------
# 4. Predictions and evaluation
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

print("Classwork 04 completed successfully.")
