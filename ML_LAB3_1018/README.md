# Machine Learning — Lab 03

**Name:** ARSHENOOR
**Registration No.:** 23-NTU-CS-1018
**Section:** BSCS 7th A
**Course:** Machine Learning Lab
**Topic:** Logistic Regression (from scratch and with scikit-learn) and Polynomial Regression

---

## Folder Structure

```
ML_LAB3_1018/
│
├── Dataset/                    All CSV files used by the scripts
│   ├── Social_Network_Ads.csv
│   ├── Position_Salaries.csv
│   ├── Telco Customer Churn.csv
│   ├── Car Price Prediction.csv
│   ├── headbrain.csv
│   ├── Admission_Predict.csv
│   ├── Medical Cost Personal Datasets.csv
│   └── 1000_Companies.csv
│
├── Classwork/                  Practice code from the lab slides
│   ├── 01_logistic_regression_scratch.py
│   ├── 02_logistic_regression_sklearn.py
│   ├── 03_polynomial_regression.py
│   └── Plots/                  Graphs produced by the scripts
│
├── Activities/                 The two activities given at the end
│   ├── activity1.py
│   ├── activity2.py
│   └── Plots/                  Graphs produced by the scripts
│
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
python 01_logistic_regression_scratch.py
```

Every script builds its dataset path relative to its own location
(`../Dataset/<file>.csv`), so no path needs to be edited before running.

---

## Classwork Summary

| # | File | Topic | Dataset | Result |
|---|------|-------|---------|--------|
| 01 | `01_logistic_regression_scratch.py` | Logistic Regression class written from scratch with the sigmoid function and gradient descent | Social_Network_Ads.csv | Accuracy = 86.25% |
| 02 | `02_logistic_regression_sklearn.py` | The same problem solved with scikit-learn, for comparison | Social_Network_Ads.csv | Accuracy = 86.25% |
| 03 | `03_polynomial_regression.py` | Polynomial Regression of degree 4, with the fitted curve and R² | Position_Salaries.csv | R² = 0.9971 |

Both logistic models reach the same accuracy on the same split, which shows the
manual gradient-descent implementation in Classwork 01 is working correctly.

In Classwork 03 a plain straight line on the same data only manages R² = 0.6715,
so the degree-4 curve is a clear improvement.

---

## Activities Summary

### Activity 1 — Customer Churn Prediction
Logistic Regression using `tenure`, `MonthlyCharges`, `Contract` and
`InternetService`. The categorical columns are one-hot encoded and the split is
stratified because the classes are imbalanced.

- **Accuracy:** 0.7913
- **Precision:** 0.6438
- **Recall:** 0.4974
- **F1-score:** 0.5612
- **ROC AUC:** 0.8268
- **Finding:** Longer tenure and longer contracts strongly reduce churn, while
  higher monthly charges raise it. Accuracy on its own is misleading because
  most customers do not churn, so recall on the churn class and the ROC AUC are
  the more honest measures.

### Activity 2 — Car Price Prediction
Linear Regression compared against Polynomial Regression of degree 2, 3 and 4,
using `enginesize`, `horsepower`, `curbweight` and `wheelbase`.

| Model | Train RMSE | Test RMSE | Train R² | Test R² |
|-------|-----------|-----------|----------|---------|
| Linear (degree 1) | 3329.93 | 4024.45 | 0.8129 | 0.7970 |
| Polynomial (degree 2) | 2995.04 | 4984.33 | 0.8486 | 0.6887 |
| Polynomial (degree 3) | 2173.46 | 4532.48 | 0.9203 | 0.7426 |
| Polynomial (degree 4) | 1275.51 | 59570.91 | 0.9725 | −43.4721 |

**Finding:** The training RMSE falls at every step while the test RMSE gets
worse, and by degree 4 the model collapses completely on unseen data. This is
overfitting — the plain linear model generalises best on this dataset.

> **Note on feature selection for Activity 2:** the activity sheet suggests
> `Kms_Driven`, `Year`, `Engine` and `Horsepower`. The provided CSV contains no
> mileage or model-year column, so the closest available specification columns
> were used instead. This is documented in a comment inside the script.

---

## Requirements

- Python 3.8 or newer
- pandas, numpy, scikit-learn, matplotlib
