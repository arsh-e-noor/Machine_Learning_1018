# ============================================================================
#  Name             : ARSHENOOR
#  Registration No. : 23-NTU-CS-1018
#  Section          : BSCS 7th A
#  Course           : Machine Learning Lab
#  Lab              : Lab 03  -  Classwork 03
#  Topic            : Polynomial Regression
#  Dataset          : Position_Salaries.csv
# ============================================================================

import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "Dataset", "Position_Salaries.csv")
PLOT_DIR = os.path.join(BASE_DIR, "Plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------------------------
data = pd.read_csv(DATA_PATH)

print("The dataset:")
print(data)

# Independent variable (Level) and dependent variable (Salary)
X = data[["Level"]].values
y = data["Salary"].values

# ---------------------------------------------------------------------------
# 2. Polynomial transformation (degree = 4 for better curve fitting)
# ---------------------------------------------------------------------------
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

# Train the polynomial regression model
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# Predictions
y_pred = poly_model.predict(X_poly)

# ---------------------------------------------------------------------------
# 3. Draw the line
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, y_pred, color="red", label="Polynomial Fit (deg=4)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.title("Polynomial Regression")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "03_polynomial_fit.png"), dpi=120)
plt.show()

# ---------------------------------------------------------------------------
# 4. Calculate the R2 score
# ---------------------------------------------------------------------------
r2 = r2_score(y, y_pred)

print("\nPOLYNOMIAL REGRESSION METRIC:")
print("R2 Score: {:.4f}".format(r2))

# A plain straight line on the same data, to show why the curve is needed
linear_model = LinearRegression()
linear_model.fit(X, y)
print("R2 Score of a simple linear fit, for comparison: {:.4f}".format(
    r2_score(y, linear_model.predict(X))
))

print("\nClasswork 03 completed successfully.")
