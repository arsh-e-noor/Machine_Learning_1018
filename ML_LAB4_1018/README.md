# Machine Learning — Lab 04

**Name:** ARSHENOOR
**Registration No.:** 23-NTU-CS-1018
**Section:** BSCS 7th A
**Course:** Machine Learning Lab
**Topic:** KNN for Credit Card Fraud Detection

---

## Folder Structure

```
ML_LAB4_1018/
│
├── Dataset/
│   └── creditcard.csv
│
├── Homework/
│   ├── activity1.py
│
└── README.md
```

---

## How to Run

```bash
pip install -r requirements.txt
cd Homework
python activity1.py
```

The script builds its dataset path relative to its own location
(`../Dataset/creditcard.csv`), so no path needs to be edited.

---

## Activity 1 — KNN for Credit Card Fraud Detection

Steps followed: load the dataset and split into X and y, train-test split,
scale with StandardScaler, apply SMOTE to fix the class imbalance, train KNN
with k = 5, then tune k using GridSearchCV. Compared using Accuracy and F1-Score.

Result for k = 5: Accuracy 0.9978, F1-Score 0.5831 (86 of 98 frauds caught).

Accuracy stays near 0.99 for every model because frauds are only 0.17% of the
data, so the F1-Score is the measure worth reading.
