# Machine Learning — Lab 05

**Name:** ARSHENOOR
**Registration No.:** 23-NTU-CS-1018
**Section:** BSCS 7th A
**Course:** Machine Learning Lab

---

## Folder Structure

```
ML_LAB5_1018/
│
├── Dataset/
│   ├── Mall_Customers.csv
│   └── brain_mri/
│       ├── yes/     (MRI images WITH a tumour)
│       └── no/      (MRI images WITHOUT a tumour)
│
├── Activities/
│   ├── activity2.py
│   ├── activity3.py
│   └── Plots/
│
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
pip install -r requirements.txt
cd Activities
python3 activity2.py
python3 activity3.py
```

Both scripts find the dataset through a relative path (`../Dataset/...`),
so nothing needs editing. Plots save into the `Plots` folder automatically.

---

## Activity 2 — SVM for Brain Tumor Detection

MRI images are converted to grayscale, resized to 64×64 and flattened into
feature vectors. Split 80:20, normalized with StandardScaler, then an SVM with
an RBF kernel (C=10, gamma='scale') is trained. Evaluated with accuracy, a
classification report and a confusion matrix, plus a grid of sample test
images showing predicted vs actual labels.

## Activity 3 — K-Means and DBSCAN on Mall Customers

Clusters Annual Income against Spending Score. Runs K-Means and DBSCAN, repeats
K-Means on scaled features to check whether scaling changed anything, uses the
Elbow Method to pick k, interprets each cluster, and finally adds a random
"LuckyNumber" column to show how a meaningless feature damages the clusters.
