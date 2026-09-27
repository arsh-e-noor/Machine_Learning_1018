# ============================================================================
#  Name             : ARSHENOOR
#  Registration No. : 23-NTU-CS-1018
#  Section          : BSCS 7th A
#  Course           : Machine Learning Lab
#  Lab              : Lab 02  -  Activity 03
#  Task             : Predict used car prices, then compare Linear Regression
#                     with Polynomial Regression of degree 2, 3 and 4.
#  Dataset          : Car Price Prediction.csv
# ============================================================================

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "Dataset", "Car Price Prediction.csv")
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
print("\nAvailable columns:")
print(list(df.columns))
print("\nFirst five rows:")
print(df.head())
print("\nMissing values in total :", df.isnull().sum().sum())

# ---------------------------------------------------------------------------
# STEP 2 - Choose the features
#
#  Note on feature selection:
#  The activity sheet suggests Kms_Driven, Year, Engine and Horsepower.
#  This particular CSV does not contain mileage or model year, so the closest
#  equivalent specification columns present in the file are used instead:
#       enginesize   -> engine capacity
#       horsepower   -> horsepower
#       curbweight   -> weight of the car
#       wheelbase    -> size of the car
#  Target = price
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2 : FEATURE SELECTION")
print("=" * 70)

features = ["enginesize", "horsepower", "curbweight", "wheelbase"]
target = "price"

X = df[features].copy()
y = df[target].copy()

print("\nFeatures used :", features)
print("Target        :", target)
print("\nCorrelation of each feature with price:")
print(df[features + [target]].corr()[target].drop(target).round(4))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples :", X_train.shape[0])
print("Testing samples  :", X_test.shape[0])

# ---------------------------------------------------------------------------
# STEP 3 - Linear Regression baseline
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3 : LINEAR REGRESSION (BASELINE)")
print("=" * 70)

linear_model = make_pipeline(StandardScaler(), LinearRegression())
linear_model.fit(X_train, y_train)

linear_train_pred = linear_model.predict(X_train)
linear_test_pred = linear_model.predict(X_test)

linear_train_rmse = np.sqrt(mean_squared_error(y_train, linear_train_pred))
linear_test_rmse = np.sqrt(mean_squared_error(y_test, linear_test_pred))
linear_train_r2 = r2_score(y_train, linear_train_pred)
linear_test_r2 = r2_score(y_test, linear_test_pred)

print("\nTraining  ->  RMSE : {:10.2f}   R2 : {:.4f}".format(linear_train_rmse, linear_train_r2))
print("Testing   ->  RMSE : {:10.2f}   R2 : {:.4f}".format(linear_test_rmse, linear_test_r2))

# ---------------------------------------------------------------------------
# STEP 4 - Polynomial Regression for degree 2, 3 and 4
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4 : POLYNOMIAL REGRESSION (DEGREE 2, 3, 4)")
print("=" * 70)

results = [
    {
        "Model": "Linear (degree 1)",
        "Train RMSE": linear_train_rmse,
        "Test RMSE": linear_test_rmse,
        "Train R2": linear_train_r2,
        "Test R2": linear_test_r2,
    }
]

for degree in [2, 3, 4]:
    poly_model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression(),
    )
    poly_model.fit(X_train, y_train)

    train_pred = poly_model.predict(X_train)
    test_pred = poly_model.predict(X_test)

    results.append(
        {
            "Model": "Polynomial (degree {})".format(degree),
            "Train RMSE": np.sqrt(mean_squared_error(y_train, train_pred)),
            "Test RMSE": np.sqrt(mean_squared_error(y_test, test_pred)),
            "Train R2": r2_score(y_train, train_pred),
            "Test R2": r2_score(y_test, test_pred),
        }
    )

comparison = pd.DataFrame(results)
print("\nPerformance comparison:\n")
print(comparison.to_string(index=False, float_format=lambda v: "{:10.4f}".format(v)))

# ---------------------------------------------------------------------------
# STEP 5 - Plot the fitted curves for the different polynomial degrees
#
#  A curve can only be drawn against one feature at a time, so 'horsepower'
#  (the feature most strongly correlated with price) is used for the plot.
# ---------------------------------------------------------------------------
single_feature = "horsepower"
X_single = df[[single_feature]].values
y_single = df[target].values

X_grid = np.linspace(X_single.min(), X_single.max(), 300).reshape(-1, 1)

plt.figure(figsize=(9, 6))
plt.scatter(X_single, y_single, color="lightseagreen", alpha=0.5, label="Actual Data")

curve_colors = {1: "#16283C", 2: "#C0392B", 3: "#8E44AD", 4: "#E67E22"}

for degree in [1, 2, 3, 4]:
    curve_model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression(),
    )
    curve_model.fit(X_single, y_single)
    y_curve = curve_model.predict(X_grid)

    label = "Linear (deg 1)" if degree == 1 else "Polynomial (deg {})".format(degree)
    plt.plot(X_grid, y_curve, color=curve_colors[degree], linewidth=2, label=label)

plt.xlabel("Horsepower")
plt.ylabel("Price")
plt.title("Activity 3 - Fitted Curves for Different Polynomial Degrees")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "activity3_polynomial_curves.png"), dpi=120)
plt.show()

# ---------------------------------------------------------------------------
# STEP 6 - Bar chart comparing the test R2 of every model
# ---------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.bar(comparison["Model"], comparison["Test R2"], color="teal")
plt.ylabel("R2 Score on the test set")
plt.title("Activity 3 - Test R2 by Model Complexity")
plt.xticks(rotation=15)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "activity3_r2_comparison.png"), dpi=120)
plt.show()

# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------
best_row = comparison.loc[comparison["Test R2"].idxmax()]

print("\n" + "=" * 70)
print("OBSERVATION")
print("=" * 70)
print(
    "\nBest performing model on the test set : {}"
    "\n   Test R2   : {:.4f}"
    "\n   Test RMSE : {:.2f}".format(
        best_row["Model"], best_row["Test R2"], best_row["Test RMSE"]
    )
)
print(
    "\nAs the degree increases, the training RMSE keeps falling because the curve\n"
    "bends to pass closer to every training point. The test RMSE, however, stops\n"
    "improving and then gets worse - this is overfitting. The model memorises the\n"
    "training data instead of learning the general pattern, so a moderate degree\n"
    "gives the best balance between underfitting and overfitting."
)

print("\nActivity 03 completed successfully.")
