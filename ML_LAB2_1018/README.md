# Machine Learning — Lab 02

**Name:** ARSHENOOR
**Registration No.:** 23-NTU-CS-1018
**Section:** BSCS 7th A
**Course:** Machine Learning Lab
**Topic:** Linear Regression, Multivariate Regression, Logistic Regression and Polynomial Regression

---

## Folder Structure

```
ML_LAB2_1018/
│
├── Dataset/                    All CSV files used by the scripts
│   ├── headbrain.csv
│   ├── Admission_Predict.csv
│   ├── Social_Network_Ads.csv
│   ├── Position_Salaries.csv
│   ├── Medical Cost Personal Datasets.csv
│   ├── Telco Customer Churn.csv
│   ├── Car Price Prediction.csv
│   └── 1000_Companies.csv
│
├── Classwork/                  Practice code from the lab slides
│   ├── 01_univariate_linear_regression.py
│   ├── 02_multivariate_linear_regression.py
│   ├── 03_logistic_regression_scratch.py
│   ├── 04_logistic_regression_sklearn.py
│   ├── 05_polynomial_regression.py
│   └── Plots/                  Graphs produced by the scripts
│
├── Activities/                 The three activities given at the end
│   ├── activity1_medical_insurance_cost.py
│   ├── activity2_telco_customer_churn.py
│   ├── activity3_car_price_prediction.py
│   └── Plots/                  Graphs produced by the scripts
│
├── requirements.txt
└── README.md
```

---

## How to Run

Install the required libraries once:

```bash
pip install -r requirements.txt
```

Then run any script directly:

```bash
cd Classwork
python 01_univariate_linear_regression.py
```

Every script builds its dataset path relative to its own location
(`../Dataset/<file>.csv`), so the code works from any machine after cloning
the repository — no path needs to be edited.

---

## Classwork Summary

| # | File | Topic | Dataset | Result |
|---|------|-------|---------|--------|
| 01 | `01_univariate_linear_regression.py` | Univariate Linear Regression built from scratch using the normal equations | headbrain.csv | R² = 0.6393, RMSE = 72.12 |
| 02 | `02_multivariate_linear_regression.py` | Multivariate Linear Regression with scikit-learn, plus feature importance | Admission_Predict.csv | R² = 0.8188, RMSE = 0.0609 |
| 03 | `03_logistic_regression_scratch.py` | Logistic Regression class written from scratch with gradient descent | Social_Network_Ads.csv | Accuracy = 88.75% |
| 04 | `04_logistic_regression_sklearn.py` | The same problem solved with scikit-learn, for comparison | Social_Network_Ads.csv | Accuracy = 88.75% |
| 05 | `05_polynomial_regression.py` | Polynomial Regression of degree 4, with R² and RMSE | Position_Salaries.csv | R² = 0.9974, RMSE = 14503.23 |

**Regression equation found in Classwork 01:**
`Brain Weight = 325.5734 + 0.2634 × Head Size`

---

## Activities Summary

### Activity 1 — Medical Insurance Cost Prediction
Linear Regression on the Medical Cost Personal dataset. The categorical
features `smoker` and `region` are encoded and the numeric features are
scaled before training.

- **Test RMSE:** 5796.77
- **Test R² Score:** 0.7836
- **Finding:** Smoking dominates every other factor — its coefficient is
  roughly 23,649, while age contributes about 3,615 and BMI about 2,036.

### Activity 2 — Telco Customer Churn Prediction
Logistic Regression using `tenure`, `MonthlyCharges`, `Contract` and
`InternetService`. The split is stratified because the classes are imbalanced.

- **Accuracy:** 0.7743
- **Precision:** 0.5927
- **Recall:** 0.4786
- **F1-score:** 0.5296
- **ROC AUC:** 0.8327
- **Finding:** Longer tenure and longer contracts strongly reduce churn, while
  fibre-optic internet raises it. Accuracy alone is misleading here since 73%
  of customers do not churn, so the ROC AUC is the more honest measure.

### Activity 3 — Car Price Prediction
Linear Regression compared against Polynomial Regression of degree 2, 3 and 4,
using `enginesize`, `horsepower`, `curbweight` and `wheelbase`.

| Model | Train RMSE | Test RMSE | Train R² | Test R² |
|-------|-----------|-----------|----------|---------|
| Linear (degree 1) | 3342.38 | 3720.84 | 0.8127 | 0.8246 |
| Polynomial (degree 2) | 2962.71 | 4377.72 | 0.8528 | 0.7572 |
| Polynomial (degree 3) | 2143.30 | 4581.49 | 0.9230 | 0.7341 |
| Polynomial (degree 4) | 1101.95 | 117200.51 | 0.9796 | −172.9962 |

**Finding:** This is a textbook demonstration of overfitting. The training RMSE
falls at every step, but the test RMSE gets steadily worse, and by degree 4 the
model collapses completely on unseen data. The plain linear model generalises
best on this dataset.

> **Note on feature selection for Activity 3:** the activity sheet suggests
> `Kms_Driven`, `Year`, `Engine` and `Horsepower`. The provided CSV contains no
> mileage or model-year column, so the closest available specification columns
> were used instead. This is documented in a comment inside the script.

---

## Requirements

- Python 3.8 or newer
- pandas, numpy, scikit-learn, matplotlib
