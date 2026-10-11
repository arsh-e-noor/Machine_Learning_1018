# ============================================================================
#  Name             : ARSHENOOR
#  Registration No. : 23-NTU-CS-1018
#  Section          : BSCS 7th A
#  Course           : Machine Learning Lab
#  Lab              : Lab 04  -  Activity 01
#  Task             : KNN for Credit Card Fraud Detection on a highly
#                     imbalanced dataset, where frauds make up less than
#                     0.2% of all samples.
#  Dataset          : creditcard.csv   (Kaggle - mlg-ulb/creditcardfraud)
# ============================================================================

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "Dataset", "creditcard.csv")
PLOT_DIR = os.path.join(BASE_DIR, "Plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# STEP 1 - Load the dataset and split it into features X and target y
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("STEP 1 : LOADING THE DATASET")
print("=" * 70)
print("\nShape of the dataset :", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nMissing values in total :", df.isnull().sum().sum())

X = df.drop("Class", axis=1)
y = df["Class"]

print("\nClass distribution:")
print("  Genuine (0) : {:>7}".format(int((y == 0).sum())))
print("  Fraud   (1) : {:>7}".format(int((y == 1).sum())))
print("  Fraud rate  : {:.4%}".format(y.mean()))
print("\nThis is the imbalance the activity warns about. A model that labelled")
print("every single transaction as genuine would already be 99.83% accurate")
print("while catching no fraud at all, so accuracy cannot be trusted here.")

# ---------------------------------------------------------------------------
# STEP 2 - Train / test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining samples :", X_train.shape[0], " (frauds:", int(y_train.sum()), ")")
print("Testing samples  :", X_test.shape[0], " (frauds:", int(y_test.sum()), ")")

# ---------------------------------------------------------------------------
# STEP 3 - Scale the features using StandardScaler
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3 : FEATURE SCALING")
print("=" * 70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeatures scaled with StandardScaler (mean = 0, standard deviation = 1).")

# ---------------------------------------------------------------------------
# STEP 4 - Apply SMOTE to handle the class imbalance
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4 : HANDLING THE IMBALANCE WITH SMOTE")
print("=" * 70)

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_train_scaled, y_train)

print("\nBefore SMOTE -> genuine: {:>7}   fraud: {:>7}".format(
    int((y_train == 0).sum()), int((y_train == 1).sum())))
print("After  SMOTE -> genuine: {:>7}   fraud: {:>7}".format(
    int((y_balanced == 0).sum()), int((y_balanced == 1).sum())))
print("\nSMOTE does not copy existing rows. It creates new synthetic fraud")
print("samples along the lines joining a fraud to its nearest fraud neighbours,")
print("so the minority class gains enough weight for KNN's vote to notice it.")

# ---------------------------------------------------------------------------
# STEP 5 - Train a KNN classifier with k = 5
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5 : KNN WITH k = 5")
print("=" * 70)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_balanced, y_balanced)

y_pred_k5 = knn.predict(X_test_scaled)

accuracy_k5 = accuracy_score(y_test, y_pred_k5)
f1_k5 = f1_score(y_test, y_pred_k5)

print("\nAccuracy : {:.4f}".format(accuracy_k5))
print("F1-Score : {:.4f}".format(f1_k5))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_k5))
print(
    "\nClassification Report:\n",
    classification_report(y_test, y_pred_k5, target_names=["Genuine", "Fraud"], digits=4),
)

# ---------------------------------------------------------------------------
# STEP 6 - Tune k using GridSearchCV
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 6 : TUNING k WITH GridSearchCV")
print("=" * 70)

SEARCH_SIZE = 20000
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(X_balanced), size=SEARCH_SIZE, replace=False)
X_search, y_search = X_balanced[sample_idx], y_balanced[sample_idx]

print("\nSearching on a stratified subsample of {:,} rows.".format(SEARCH_SIZE))

param_grid = {"n_neighbors": [1, 3, 5, 7, 9]}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3, scoring="f1")
grid.fit(X_search, y_search)

print("\nCross-validated F1-Score for each k:")
for k, score in zip(param_grid["n_neighbors"], grid.cv_results_["mean_test_score"]):
    print("  k = {:<3} ->  F1 = {:.4f}".format(k, score))

best_k = grid.best_params_["n_neighbors"]
print("\nBest value of k :", best_k)

# Refit the winning k on the complete balanced training set
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_balanced, y_balanced)

y_pred_best = best_knn.predict(X_test_scaled)

accuracy_best = accuracy_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best)

print("\nAccuracy : {:.4f}".format(accuracy_best))
print("F1-Score : {:.4f}".format(f1_best))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_best))
print(
    "\nClassification Report:\n",
    classification_report(y_test, y_pred_best, target_names=["Genuine", "Fraud"], digits=4),
)

# ---------------------------------------------------------------------------
# STEP 7 - Compare model performance using Accuracy and F1-Score
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 7 : PERFORMANCE COMPARISON")
print("=" * 70)

comparison = pd.DataFrame(
    [
        {"Model": "KNN (k = 5)", "Accuracy": accuracy_k5, "F1-Score": f1_k5},
        {"Model": "KNN (tuned, k = {})".format(best_k),
         "Accuracy": accuracy_best, "F1-Score": f1_best},
    ]
)

print()
print(comparison.to_string(index=False, float_format=lambda v: "{:10.4f}".format(v)))

# ---------------------------------------------------------------------------
# STEP 8 - Confusion matrix of the tuned model
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred_best)

fig, ax = plt.subplots(figsize=(6, 5))
display = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["Genuine", "Fraud"]
)
display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
ax.set_title("Activity 1 - Confusion Matrix (tuned KNN, k = {})".format(best_k))
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "activity1_confusion_matrix.png"), dpi=120)
plt.show()

# ---------------------------------------------------------------------------
# STEP 9 - F1-Score against every value of k that was tried
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(param_grid["n_neighbors"], grid.cv_results_["mean_test_score"],
         marker="o", color="teal", linewidth=2)
plt.axvline(best_k, color="red", linestyle="--", linewidth=1.5,
            label="Best k = {}".format(best_k))
plt.xlabel("k  (number of neighbours)")
plt.ylabel("Cross-validated F1-Score")
plt.title("Activity 1 - F1-Score vs k")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "activity1_f1_vs_k.png"), dpi=120)
plt.show()

# ---------------------------------------------------------------------------
# STEP 10 - Accuracy and F1-Score side by side
# ---------------------------------------------------------------------------
labels = comparison["Model"].tolist()
positions = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(positions - width / 2, comparison["Accuracy"], width,
        label="Accuracy", color="#16283C")
plt.bar(positions + width / 2, comparison["F1-Score"], width,
        label="F1-Score", color="teal")
plt.xticks(positions, labels)
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Activity 1 - Accuracy vs F1-Score")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "activity1_accuracy_vs_f1.png"), dpi=120)
plt.show()

# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("OBSERVATION")
print("=" * 70)
print(
    "\nThe two scores tell completely different stories, and that is the whole\n"
    "point of this activity. Accuracy sits around 0.99 for every model tried,\n"
    "including ones that miss a large share of the frauds, because genuine\n"
    "transactions outnumber frauds by roughly 580 to 1. The F1-Score moves a\n"
    "great deal between models and is the honest measure here, since it only\n"
    "rises when frauds are actually caught without flagging large numbers of\n"
    "genuine customers along the way.\n\n"
    "SMOTE is what makes KNN usable at all on this data. Without it almost\n"
    "every test point is surrounded by genuine neighbours, so the majority\n"
    "vote returns 'genuine' nearly every time. Balancing the training set\n"
    "gives the fraud class enough presence for the vote to go its way.\n\n"
    "A small k works best. Raising k widens the neighbourhood until the\n"
    "genuine majority outvotes the frauds again, which is exactly the problem\n"
    "SMOTE was brought in to solve."
)

print("\nActivity 01 completed successfully.")
