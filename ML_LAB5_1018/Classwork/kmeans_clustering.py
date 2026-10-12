# ============================================================================
#  Name             : ARSHENOOR
#  Registration No. : 23-NTU-CS-1018
#  Section          : BSCS 7th A
#  Course           : Machine Learning Lab
#  Lab              : Lab 04  -  K-Means Clustering
#  Dataset          : mushroomss.csv
#
#  Notes:
#   * seaborn is NOT needed, everything is drawn with matplotlib
#   * works whether or not the CSV has the 'class' label column
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------------------------
mushroom_data = pd.read_csv("mushroomss.csv")

print("Dataset Preview:")
print(mushroom_data.head())

print("\nMissing Values in Each Column:")
print(mushroom_data.isnull().sum())

print(f"\nDataset shape: {mushroom_data.shape}")

# Find the label column. Different copies of this dataset name it differently,
# and some copies drop it entirely.
label_col = None
for candidate in ["class", "Class", "type", "label", "edible", "poisonous"]:
    if candidate in mushroom_data.columns:
        label_col = candidate
        break

if label_col:
    print(f"Label column found: '{label_col}'")
    print(f"Target distribution:\n{mushroom_data[label_col].value_counts()}")
else:
    print("No label column in this CSV - Part A will be skipped,")
    print("only the unsupervised clustering in Part B will run.")


def countplot(ax, column, hue=None, title=""):
    """Plain-matplotlib stand-in for sns.countplot."""
    if hue is None:
        counts = mushroom_data[column].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values, color="#4C72B0")
    else:
        table = mushroom_data.groupby([column, hue]).size().unstack(fill_value=0)
        categories = table.index.astype(str)
        positions = np.arange(len(categories))
        width = 0.8 / len(table.columns)
        for i, level in enumerate(table.columns):
            ax.bar(positions + i * width, table[level].values, width, label=str(level))
        ax.set_xticks(positions + width * (len(table.columns) - 1) / 2)
        ax.set_xticklabels(categories)
        ax.legend(title=hue, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("count")


# ===========================================================================
#  PART A  -  K-Means compared against the true labels
#             (runs only if the CSV actually has a label column)
# ===========================================================================
if label_col:

    # ----- Exploratory data analysis -----
    plt.figure(figsize=(15, 10))
    countplot(plt.subplot(2, 3, 1), label_col, title='Target Distribution')
    countplot(plt.subplot(2, 3, 2), 'cap-shape', hue=label_col, title='Cap Shape vs Class')
    countplot(plt.subplot(2, 3, 3), 'cap-color', hue=label_col, title='Cap Color vs Class')
    countplot(plt.subplot(2, 3, 4), 'population', title='Population Distribution')
    countplot(plt.subplot(2, 3, 5), 'habitat', title='Habitat Distribution')
    countplot(plt.subplot(2, 3, 6), 'odor', hue=label_col, title='Odor vs Class')
    plt.tight_layout()
    plt.savefig("kmeans_eda.png", dpi=120)
    plt.show()

    # ----- Preprocessing -----
    mushroom_df = mushroom_data.dropna()
    mushroom_features = mushroom_df.drop(label_col, axis=1)
    mushroom_labels = mushroom_df[label_col]

    encoder = OrdinalEncoder()
    mushroom_prepared = encoder.fit_transform(mushroom_features)
    mushroom_prep_df = pd.DataFrame(mushroom_prepared,
                                    columns=mushroom_features.columns)

    # ----- KMeans with 2 clusters -----
    k_clust = KMeans(n_clusters=2, random_state=42, n_init=10)
    k_clust.fit(mushroom_prep_df)
    k_labels = k_clust.labels_

    # ----- Evaluation -----
    true_labels = LabelEncoder().fit_transform(mushroom_labels)

    accuracy = accuracy_score(true_labels, k_labels)
    # Cluster numbering is arbitrary, so if the clusters came out swapped the
    # accuracy would look terrible. Flip them and keep whichever is better.
    if accuracy < 0.5:
        k_labels = 1 - k_labels
        accuracy = accuracy_score(true_labels, k_labels)

    print(f"\nKMeans Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(true_labels, k_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, format(v, 'd'), ha='center', va='center', fontsize=14,
                color='white' if v > cm.max() / 2 else 'black')
    ax.set_xticks([0, 1], ['Cluster 0', 'Cluster 1'])
    ax.set_yticks([0, 1], ['Edible', 'Poisonous'])
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Clusters')
    ax.set_ylabel('True Labels')
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig("kmeans_confusion_matrix.png", dpi=120)
    plt.show()

    print("\nClassification Report:")
    print(classification_report(true_labels, k_labels,
                                target_names=['Edible', 'Poisonous']))


# ===========================================================================
#  PART B  -  Unsupervised: 3 clusters visualised with PCA
# ===========================================================================
data = pd.read_csv("mushroomss.csv")

# Encode every text column.
# The slide tests `dtype == 'object'`. On newer pandas text columns are no
# longer 'object', so that test skips everything and KMeans then crashes with
# "could not convert string to float". is_string_dtype covers both versions.
label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object' or pd.api.types.is_string_dtype(data[column]):
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(data)

print("\nCluster Centers:\n", kmeans.cluster_centers_)
print("\nCluster Counts:\n", data['Cluster'].value_counts())

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data.drop('Cluster', axis=1))

plt.figure(figsize=(8, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1],
            c=data['Cluster'], cmap='viridis', s=50)
plt.title('K-Means Clustering Visualization')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.tight_layout()
plt.savefig("kmeans_pca_clusters.png", dpi=120)
plt.show()

print("\nDone. Plots saved as PNG files in this folder.")