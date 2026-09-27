# ============================================================================
#  Name             : ARSHENOOR
#  Registration No. : 23-NTU-CS-1018
#  Section          : BSCS 7th A
#  Course           : Machine Learning Lab
#  Lab              : Lab 02  -  Activity 02
#  Task             : Predict whether a customer will churn (leave the company)
#                     using Logistic Regression.
#  Dataset          : Telco Customer Churn.csv
# ============================================================================

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "Dataset", "Telco Customer Churn.csv")
PLOT_DIR = os.path.join(BASE_DIR, "Plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# STEP 1 - Load the dataset
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("STEP 1 : LOADING THE DATASET")
print("=" * 70)
print("\nShape of the dataset :", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nDistribution of the target variable (Churn):")
print(df["Churn"].value_counts())

# ---------------------------------------------------------------------------
# STEP 2 - Preprocessing the categorical variables
#   Features : tenure, MonthlyCharges, Contract, InternetService
#   Target   : Churn
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2 : PREPROCESSING")
print("=" * 70)

numeric_features = ["tenure", "MonthlyCharges"]
categorical_features = ["Contract", "InternetService"]

X = df[numeric_features + categorical_features].copy()

# Target: Yes / No  ->  1 / 0
y = df["Churn"].map({"No": 0, "Yes": 1})

print("\nCategories found in the categorical features:")
for column in categorical_features:
    print("  {:<18} : {}".format(column, list(X[column].unique())))

# One-Hot Encoding for the unordered categorical features
X = pd.get_dummies(X, columns=categorical_features, drop_first=True, dtype=int)

print("\nFeature columns after encoding:")
print(list(X.columns))
print("\nFirst five rows after encoding:")
print(X.head())

# --- Train / test split -------------------------------------------------
# stratify keeps the same churn ratio in both sets, which matters because
# the classes are imbalanced (roughly 73% stay, 27% churn).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Scale the numerical features ---------------------------------------
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

print("\nTraining samples :", X_train_scaled.shape[0])
print("Testing samples  :", X_test_scaled.shape[0])

# ---------------------------------------------------------------------------
# STEP 3 - Train the Logistic Regression model
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3 : TRAINING THE LOGISTIC REGRESSION MODEL")
print("=" * 70)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

print("\nLearned coefficients:")
for feature, coefficient in zip(X_train_scaled.columns, model.coef_[0]):
    print("  {:<28} : {: .4f}".format(feature, coefficient))
print("  {:<28} : {: .4f}".format("Intercept", model.intercept_[0]))

# ---------------------------------------------------------------------------
# STEP 4 - Evaluation: Accuracy, Precision, Recall, F1-score
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4 : MODEL EVALUATION")
print("=" * 70)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]  # probability of churn

print("\nAccuracy  : {:.4f}".format(accuracy_score(y_test, y_pred)))
print("Precision : {:.4f}".format(precision_score(y_test, y_pred)))
print("Recall    : {:.4f}".format(recall_score(y_test, y_pred)))
print("F1-score  : {:.4f}".format(f1_score(y_test, y_pred)))
print("ROC AUC   : {:.4f}".format(roc_auc_score(y_test, y_prob)))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(
    "\nClassification Report:\n",
    classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]),
)

# ---------------------------------------------------------------------------
# STEP 5 - Plot the Confusion Matrix
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))
display = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["No Churn", "Churn"]
)
display.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Activity 2 - Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "activity2_confusion_matrix.png"), dpi=120)
plt.show()

# ---------------------------------------------------------------------------
# STEP 6 - Plot the ROC Curve
# ---------------------------------------------------------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="teal", linewidth=2, label="Logistic Regression (AUC = {:.3f})".format(auc))
plt.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=1.5, label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Activity 2 - ROC Curve for Customer Churn")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "activity2_roc_curve.png"), dpi=120)
plt.show()

# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("OBSERVATION")
print("=" * 70)
print(
    "\nTenure has a strong negative coefficient, so the longer a customer has\n"
    "been with the company the less likely they are to churn. A two-year or\n"
    "one-year contract also pushes the probability down, while fibre-optic\n"
    "internet and higher monthly charges push it up.\n\n"
    "Accuracy alone is misleading here because about 73% of customers do not\n"
    "churn - a model that always predicted 'No Churn' would already score 73%.\n"
    "Recall on the churn class is therefore the more useful number, and the\n"
    "ROC AUC of about {:.2f} shows the model separates the two classes well.".format(auc)
)

print("\nActivity 02 completed successfully.")
