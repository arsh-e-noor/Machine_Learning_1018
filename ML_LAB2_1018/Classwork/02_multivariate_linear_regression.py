# ============================================================================
#  Name             : ARSHENOOR
#  Registration No. : 23-NTU-CS-1018
#  Section          : BSCS 7th A
#  Course           : Machine Learning Lab
#  Lab              : Lab 02  -  Classwork 02
#  Topic            : Multivariate Linear Regression (scikit-learn)
#  Dataset          : Admission_Predict.csv
# ============================================================================

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "Dataset", "Admission_Predict.csv")
PLOT_DIR = os.path.join(BASE_DIR, "Plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

# Some column names in this CSV carry trailing spaces
# (for example "Chance of Admit " and "LOR "), so they are cleaned first.
df.columns = df.columns.str.strip()
print("Columns in the dataset:", list(df.columns))

# Drop the serial number column - it is only an index, not a feature
df.drop("Serial No.", axis=1, inplace=True)

# Separate the target variable from the features
y = df["Chance of Admit"]
df.drop("Chance of Admit", axis=1, inplace=True)

print("\nFirst five rows of the feature set:")
print(df.head())

# ---------------------------------------------------------------------------
# 2. Simple (univariate) regression using only the GRE Score
# ---------------------------------------------------------------------------
simple_lr = LinearRegression()
simple_lr.fit(df[["GRE Score"]], y)

plt.figure(figsize=(8, 5))
plt.scatter(df["GRE Score"], y, color="green", label="Actual Data", alpha=0.5)
plt.plot(
    df["GRE Score"],
    simple_lr.predict(df[["GRE Score"]]),
    color="red",
    linewidth=3,
    label="Regression Line",
)
plt.xlabel("GRE Score")
plt.ylabel("Admission chance")
plt.title("GRE Score vs Admission chance")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "02_gre_vs_admission.png"), dpi=120)
plt.show()

# ---------------------------------------------------------------------------
# 3. Train / test split for the multivariate model
# ---------------------------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", x_train.shape[0])
print("Testing samples :", x_test.shape[0])

# ---------------------------------------------------------------------------
# 4. Train the multivariate linear regression model
# ---------------------------------------------------------------------------
lr = LinearRegression()
lr.fit(x_train, y_train)

coefficients = lr.coef_
features = df.columns

print("\nLearned coefficients (feature weights):")
for feature, coefficient in zip(features, coefficients):
    print("  {:<20} : {: .6f}".format(feature, coefficient))
print("  {:<20} : {: .6f}".format("Intercept", lr.intercept_))

# ---------------------------------------------------------------------------
# 5. Feature importance plot
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.barh(features, coefficients, color="teal")
plt.xlabel("Coefficient Value (Weight)")
plt.title("Feature Importance in Multivariate Linear Regression")
plt.axvline(x=0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "02_feature_importance.png"), dpi=120)
plt.show()

# ---------------------------------------------------------------------------
# 6. Predictions and evaluation
# ---------------------------------------------------------------------------
pred = lr.predict(x_test)

rmse = np.sqrt(metrics.mean_squared_error(y_test, pred))
r2 = metrics.r2_score(y_test, pred)

print("\nModel Evaluation on the test set")
print("RMSE     :", rmse)
print("R2 Score :", r2)

print("\nClasswork 02 completed successfully.")
