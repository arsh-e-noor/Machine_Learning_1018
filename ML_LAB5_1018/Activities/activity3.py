# ============================================================================
#  Name             : ARSHENOOR
#  Registration No. : 23-NTU-CS-1018
#  Section          : BSCS 7th A
#  Course           : Machine Learning Lab
#  Lab              : Lab 05  -  Activity 03
#  Task             : K-Means and DBSCAN clustering on mall customers,
#                     the effect of scaling, the Elbow Method, and what a
#                     fake random feature does to the clusters.
#  Dataset          : Mall_Customers.csv
# ============================================================================

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "Dataset", "Mall_Customers.csv")
PLOT_DIR = os.path.join(BASE_DIR, "Plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# STEP 1 - Load the dataset and display it
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("STEP 1 : LOADING THE DATASET")
print("=" * 70)
print("\nShape :", df.shape)
print("\nFirst ten rows:")
print(df.head(10))
print("\nColumn names:", list(df.columns))
print("\nSummary:")
print(df.describe())

# The two columns the activity asks to cluster on. Different copies of this
# file name them slightly differently, so they are matched loosely.
income_col = [c for c in df.columns if "Income" in c][0]
score_col = [c for c in df.columns if "Score" in c][0]
print("\nUsing columns :", income_col, "and", score_col)

X = df[[income_col, score_col]].values

# ---------------------------------------------------------------------------
# STEP 2 - K-Means and DBSCAN on the raw (unscaled) data
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2 : K-MEANS AND DBSCAN ON UNSCALED DATA")
print("=" * 70)

kmeans_raw = KMeans(n_clusters=5, random_state=42, n_init=10)
labels_kmeans_raw = kmeans_raw.fit_predict(X)

dbscan_raw = DBSCAN(eps=8, min_samples=5)
labels_dbscan_raw = dbscan_raw.fit_predict(X)

n_noise = int((labels_dbscan_raw == -1).sum())
n_clusters_db = len(set(labels_dbscan_raw)) - (1 if -1 in labels_dbscan_raw else 0)

print("\nK-Means cluster sizes :")
print(pd.Series(labels_kmeans_raw).value_counts().sort_index().to_string())
print("\nDBSCAN found {} clusters and marked {} points as noise.".format(
    n_clusters_db, n_noise))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
axes[0].scatter(X[:, 0], X[:, 1], c=labels_kmeans_raw, cmap="viridis", s=50)
axes[0].scatter(kmeans_raw.cluster_centers_[:, 0], kmeans_raw.cluster_centers_[:, 1],
                c="red", marker="X", s=200, label="Centroids")
axes[0].set_title("K-Means (unscaled)")
axes[0].legend()
axes[1].scatter(X[:, 0], X[:, 1], c=labels_dbscan_raw, cmap="viridis", s=50)
axes[1].set_title("DBSCAN (unscaled) - dark points are noise")
for ax in axes:
    ax.set_xlabel(income_col)
    ax.set_ylabel(score_col)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "activity3_kmeans_vs_dbscan.png"), dpi=120)
plt.show()

# ---------------------------------------------------------------------------
# STEP 3 - Scale the features and run K-Means again
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3 : K-MEANS AFTER SCALING")
print("=" * 70)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans_scaled = KMeans(n_clusters=5, random_state=42, n_init=10)
labels_kmeans_scaled = kmeans_scaled.fit_predict(X_scaled)

print("\nCluster sizes after scaling:")
print(pd.Series(labels_kmeans_scaled).value_counts().sort_index().to_string())

# ---------------------------------------------------------------------------
# STEP 4 - Did scaling change the clusters?
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4 : DID SCALING CHANGE THE CLUSTERS?")
print("=" * 70)

# Cluster numbers are arbitrary, so compare how points are grouped together
# rather than comparing the label numbers directly.
same_group_before = labels_kmeans_raw[:, None] == labels_kmeans_raw[None, :]
same_group_after = labels_kmeans_scaled[:, None] == labels_kmeans_scaled[None, :]
agreement = (same_group_before == same_group_after).mean()

print("\nAgreement between the two groupings : {:.2%}".format(agreement))
print(
    "\nBoth columns are already on a similar range here - income runs roughly\n"
    "15-140 and the score 1-100 - so scaling moves only a few border points.\n"
    "Scaling matters far more when one column is in thousands and another is\n"
    "in single digits, because K-Means measures plain distance and the larger\n"
    "column would otherwise decide the clusters on its own."
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
axes[0].scatter(X[:, 0], X[:, 1], c=labels_kmeans_raw, cmap="viridis", s=50)
axes[0].set_title("K-Means before scaling")
axes[1].scatter(X[:, 0], X[:, 1], c=labels_kmeans_scaled, cmap="viridis", s=50)
axes[1].set_title("K-Means after scaling")
for ax in axes:
    ax.set_xlabel(income_col)
    ax.set_ylabel(score_col)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "activity3_scaling_comparison.png"), dpi=120)
plt.show()

# ---------------------------------------------------------------------------
# STEP 5 - Elbow Method to find the best k
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5 : ELBOW METHOD")
print("=" * 70)

k_range = range(1, 11)
inertias = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

print("\n  k    inertia (within-cluster sum of squares)")
for k, inertia in zip(k_range, inertias):
    print("  {:<4} {:>14,.0f}".format(k, inertia))

plt.figure(figsize=(8, 5))
plt.plot(list(k_range), inertias, marker="o", color="teal", linewidth=2)
plt.axvline(5, color="red", linestyle="--", linewidth=1.5, label="Elbow at k = 5")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Activity 3 - Elbow Method")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "activity3_elbow_method.png"), dpi=120)
plt.show()

print("\nThe curve drops steeply up to k = 5 and then flattens, so k = 5 is the")
print("elbow. Adding more clusters past that buys very little.")

# ---------------------------------------------------------------------------
# STEP 6 - Interpret each cluster
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6 : INTERPRETING EACH CLUSTER")
print("=" * 70)

df["Cluster"] = labels_kmeans_raw
profile = df.groupby("Cluster")[[income_col, score_col]].mean().round(1)
profile["Customers"] = df.groupby("Cluster").size()

print()
print(profile.to_string())

overall_income = df[income_col].mean()
overall_score = df[score_col].mean()

print("\nWhat each cluster means:")
for cluster_id, row in profile.iterrows():
    income_level = "high" if row[income_col] > overall_income else "low"
    spending_level = "high" if row[score_col] > overall_score else "low"

    meaning = {
        ("high", "high"): "earns well and spends well - the best customers to target",
        ("high", "low"): "earns well but spends little - worth winning over",
        ("low", "high"): "earns less but still spends a lot - loyal shoppers",
        ("low", "low"): "earns less and spends little - least profitable group",
    }[(income_level, spending_level)]

    print("  Cluster {} ({:>3} customers) : {}".format(
        cluster_id, int(row["Customers"]), meaning))

# ---------------------------------------------------------------------------
# STEP 7 - Add a fake "LuckyNumber" feature and see what it does
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 7 : ADDING A FAKE 'LuckyNumber' FEATURE")
print("=" * 70)

rng = np.random.default_rng(42)
df["LuckyNumber"] = rng.integers(0, 101, size=len(df))

X_fake = df[[income_col, score_col, "LuckyNumber"]].values
kmeans_fake = KMeans(n_clusters=5, random_state=42, n_init=10)
labels_fake = kmeans_fake.fit_predict(X_fake)

same_before = labels_kmeans_raw[:, None] == labels_kmeans_raw[None, :]
same_after = labels_fake[:, None] == labels_fake[None, :]
agreement_fake = (same_before == same_after).mean()

moved = int((pd.crosstab(labels_kmeans_raw, labels_fake).max(axis=1).sum()))
print("\nAgreement with the original clustering : {:.2%}".format(agreement_fake))
print("Customers still grouped the same way   : {} of {}".format(moved, len(df)))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
axes[0].scatter(X[:, 0], X[:, 1], c=labels_kmeans_raw, cmap="viridis", s=50)
axes[0].set_title("Without LuckyNumber")
axes[1].scatter(X[:, 0], X[:, 1], c=labels_fake, cmap="viridis", s=50)
axes[1].set_title("With the fake LuckyNumber feature")
for ax in axes:
    ax.set_xlabel(income_col)
    ax.set_ylabel(score_col)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "activity3_luckynumber_effect.png"), dpi=120)
plt.show()

print(
    "\nExplanation:\n"
    "LuckyNumber is pure noise - it carries no information about any customer.\n"
    "But K-Means cannot tell a meaningful column from a meaningless one. It\n"
    "measures distance across every column it is given, so this random column\n"
    "pulls points around and splits groups that genuinely belonged together.\n"
    "Its range (0-100) is as wide as the spending score, so it gets just as\n"
    "much say in the result.\n\n"
    "The lesson: clustering has no target to check itself against, so adding\n"
    "junk features silently damages the clusters. Features must be chosen with\n"
    "care rather than thrown in because they are available."
)

print("\nPlots saved in the Plots folder.")
print("Activity 03 completed successfully.")
