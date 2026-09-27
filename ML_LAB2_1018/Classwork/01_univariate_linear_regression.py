# ============================================================================
#  Name             : ARSHENOOR
#  Registration No. : 23-NTU-CS-1018
#  Section          : BSCS 7th A
#  Course           : Machine Learning Lab
#  Lab              : Lab 02  -  Classwork 01
#  Topic            : Univariate Linear Regression (from scratch)
#  Dataset          : headbrain.csv
# ============================================================================

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative, so the script runs from anywhere after cloning the repo)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "Dataset", "headbrain.csv")
PLOT_DIR = os.path.join(BASE_DIR, "Plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

print("First five rows of the dataset:")
print(df.head())
print("\nShape of the dataset:", df.shape)

# ---------------------------------------------------------------------------
# 2. Computing X and y
# ---------------------------------------------------------------------------
X = df["Head Size(cm^3)"].values
y = df["Brain Weight(grams)"].values

# ---------------------------------------------------------------------------
# 3. Calculate the basic summations needed by the normal equations
# ---------------------------------------------------------------------------
X_sum = np.sum(X)
X_squared_sum = np.sum(X * X)
n = len(X)

print("\nSum of X:", X_sum)
print("Sum of X squared:", X_squared_sum)
print("Length of X:", n)

# ---------------------------------------------------------------------------
# 4. Estimate the parameters  w1 (slope) and w0 (intercept)
#
#         n * sum(X*y) - sum(X) * sum(y)
#    w1 = ------------------------------
#            n * sum(X*X) - (sum(X))^2
#
#         sum(y) - w1 * sum(X)
#    w0 = --------------------
#                  n
# ---------------------------------------------------------------------------
numer = n * np.sum(X * y) - np.sum(X) * np.sum(y)
denom = n * np.sum(X * X) - (np.sum(X)) ** 2

w1 = numer / denom
w0 = (np.sum(y) - w1 * (np.sum(X))) / n

print("\nEstimated Parameters")
print("w1 (slope)     =", w1)
print("w0 (intercept) =", w0)
print("Regression equation:  y_hat = {:.4f} + {:.4f} * x".format(w0, w1))

# ---------------------------------------------------------------------------
# 5. Plot the regression line over the scatter plot
# ---------------------------------------------------------------------------
max_x = np.max(X)
min_x = np.min(X)

# Calculating line values x and y
x1 = np.linspace(min_x, max_x)
y1 = w0 + w1 * x1

plt.figure(figsize=(8, 5))
plt.plot(x1, y1, color="red", label="Regression Line")
plt.scatter(X, y, c="green", label="Scatter Plot")
plt.xlabel("Head Size in cm3")
plt.ylabel("Brain Weight in grams")
plt.title("Univariate Linear Regression - Head Size vs Brain Weight")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "01_regression_line.png"), dpi=120)
plt.show()

# ---------------------------------------------------------------------------
# 6. Root Mean Squared Error
# ---------------------------------------------------------------------------
rmse = 0
for i in range(n):
    y_pred = w0 + w1 * X[i]
    rmse += (y[i] - y_pred) ** 2
rmse = np.sqrt(rmse / n)

print("\nRMSE =", rmse)

# ---------------------------------------------------------------------------
# 7. R2 Score
#        R2 = 1 - (SS_res / SS_tot)
# ---------------------------------------------------------------------------
ss_tot = 0
ss_res = 0
y_mean = np.mean(y)

for i in range(n):
    y_pred = w0 + w1 * X[i]
    ss_tot += (y[i] - y_mean) ** 2
    ss_res += (y[i] - y_pred) ** 2

r2 = 1 - (ss_res / ss_tot)
print("R2 Score =", r2)

print("\nClasswork 01 completed successfully.")
