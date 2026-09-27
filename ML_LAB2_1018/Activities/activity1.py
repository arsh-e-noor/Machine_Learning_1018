# ============================================================================
#  Name             : ARSHENOOR
#  Registration No. : 23-NTU-CS-1018
#  Section          : BSCS 7th A
#  Course           : Machine Learning Lab
#  Lab              : Lab 02  -  Activity 01
#  Task             : Predict a person's medical insurance cost from their
#                     personal details using Linear Regression.
#  Dataset          : Medical Cost Personal Datasets.csv
# ============================================================================

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    BASE_DIR, "..", "Dataset", "Medical Cost Personal Datasets.csv"
)
PLOT_DIR = os.path.join(BASE_DIR, "Plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# STEP 1 - Load and explore the dataset
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("STEP 1 : LOADING AND EXPLORING THE DATASET")
print("=" * 70)
print("\nShape of the dataset :", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nColumn data types:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nStatistical summary of the numeric columns:")
print(df.describe())

# ---------------------------------------------------------------------------
# STEP 2 - Preprocessing
#   Features : age, bmi, children, smoker, region
#   Target   : charges
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2 : PREPROCESSING")
print("=" * 70)

numeric_features = ["age", "bmi", "children"]
categorical_features = ["smoker", "region"]
target_column = "charges"

X = df[numeric_features + categorical_features].copy()
y = df[target_column].copy()

# --- Encoding the categorical features ---------------------------------
# 'smoker' has only two values, so a simple binary mapping is enough.
X["smoker"] = X["smoker"].map({"no": 0, "yes": 1})

# 'region' has four unordered values, so One-Hot Encoding is used.
# drop_first=True avoids the dummy variable trap.
X = pd.get_dummies(X, columns=["region"], drop_first=True, dtype=int)

print("\nFeature columns after encoding:")
print(list(X.columns))
print("\nFirst five rows after encoding:")
print(X.head())

# --- Train / test split -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Scaling the numerical features -------------------------------------
# Only the genuinely numeric columns are scaled; the 0/1 encoded columns
# are already on a comparable scale.
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

print("\nTraining samples :", X_train_scaled.shape[0])
print("Testing samples  :", X_test_scaled.shape[0])

# ---------------------------------------------------------------------------
# STEP 3 - Train the Linear Regression model
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3 : TRAINING THE LINEAR REGRESSION MODEL")
print("=" * 70)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("\nLearned coefficients:")
for feature, coefficient in zip(X_train_scaled.columns, model.coef_):
    print("  {:<22} : {: 12.4f}".format(feature, coefficient))
print("  {:<22} : {: 12.4f}".format("Intercept", model.intercept_))

# ---------------------------------------------------------------------------
# STEP 4 - Evaluate using RMSE and R2 score
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4 : MODEL EVALUATION")
print("=" * 70)

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nTraining set  ->  RMSE : {:.4f}   R2 Score : {:.4f}".format(train_rmse, train_r2))
print("Testing  set  ->  RMSE : {:.4f}   R2 Score : {:.4f}".format(test_rmse, test_r2))

# ---------------------------------------------------------------------------
# STEP 5 - Plot Predicted vs Actual costs
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color="teal", alpha=0.6, label="Predictions")

# A perfect model would place every point on this diagonal line
lower = min(y_test.min(), y_test_pred.min())
upper = max(y_test.max(), y_test_pred.max())
plt.plot([lower, upper], [lower, upper], color="red", linewidth=2, label="Perfect Fit")

plt.xlabel("Actual Insurance Cost (charges)")
plt.ylabel("Predicted Insurance Cost")
plt.title("Activity 1 - Predicted vs Actual Medical Insurance Cost")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "activity1_predicted_vs_actual.png"), dpi=120)
plt.show()

# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("OBSERVATION")
print("=" * 70)
print(
    "\nSmoking is by far the strongest driver of insurance cost - its coefficient\n"
    "is much larger than every other feature. Age and BMI also raise the cost,\n"
    "while the region makes only a small difference. The R2 score of about\n"
    "{:.2f} means the model explains roughly {:.0f}% of the variation in charges.\n"
    "The remaining error mostly comes from the group of high-cost smokers, which\n"
    "a straight-line model cannot separate from the rest.".format(
        test_r2, test_r2 * 100
    )
)

print("\nActivity 01 completed successfully.")
