# ============================================================================
#  Name             : ARSHENOOR
#  Registration No. : 23-NTU-CS-1018
#  Section          : BSCS 7th A
#  Course           : Machine Learning Lab
#  Lab              : Lab 03  -  Classwork 01
#  Topic            : Logistic Regression implemented from scratch
#  Dataset          : Social_Network_Ads.csv
# ============================================================================

import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "Dataset", "Social_Network_Ads.csv")

# ---------------------------------------------------------------------------
# 1. Load the dataset from a CSV file
# ---------------------------------------------------------------------------
data = pd.read_csv(DATA_PATH)

print("First five rows of the dataset:")
print(data.head())

# ---------------------------------------------------------------------------
# 2. Preprocessing - encode the categorical variable 'Gender'
# ---------------------------------------------------------------------------
data = pd.get_dummies(data, columns=["Gender"], drop_first=True)

# Define features and target variable
X = data[["Gender_Male", "Age", "EstimatedSalary"]].values
y = data["Purchased"].values  # Target variable

# ---------------------------------------------------------------------------
# 3. Split the dataset into training and testing sets
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the features so that gradient descent converges properly
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ---------------------------------------------------------------------------
# 4. Logistic Regression implementation
# ---------------------------------------------------------------------------
class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.lr = learning_rate
        self.iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for i in range(self.iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            # Update weights and bias by calculating the gradient
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return [1 if i > 0.5 else 0 for i in self.sigmoid(linear_model)]


# ---------------------------------------------------------------------------
# 5. Train the model
# ---------------------------------------------------------------------------
model = LogisticRegression(learning_rate=0.1, num_iterations=1000)
model.fit(X_train, y_train)

print("\nLearned weights :", model.weights)
print("Learned bias    :", model.bias)

# ---------------------------------------------------------------------------
# 6. Evaluate the model
# ---------------------------------------------------------------------------
predictions = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))

print("Classwork 01 completed successfully.")
