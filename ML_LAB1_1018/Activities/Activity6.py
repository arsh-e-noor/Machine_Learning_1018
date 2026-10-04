
#23-NTU-CS-1018
#ARSHE NOOR
#BSCS - 7TH A
# --------------------
# BREAST CANCER DATA
# Data preparation, outliers, transformation, scaling,
# PCA, SMOTE, and classification
# --------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from imblearn.over_sampling import SMOTE


# --------------------
# 1. IMPORT THE DATASET
# --------------------

data = load_breast_cancer()


# Display dataset details
print("==========================================")
print("DATASET INFORMATION")
print("==========================================")

print("\ndata.keys():")
print(data.keys())

print("\nTarget names:")
print(data.target_names)

print("\nNumber of samples:")
print(data.data.shape[0])

print("\nNumber of features:")
print(data.data.shape[1])

print("\nFeature names:")
print(data.feature_names)

print("\nShape of data.data:")
print(data.data.shape)

print("\nShape of data.target:")
print(data.target.shape)


# --------------------
# 2. BUILD THE PANDAS DATAFRAME
# --------------------

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

# Append the target column
df["target"] = data.target

print("\n==========================================")
print("FIRST 5 ROWS")
print("==========================================")

print(df.head())


# --------------------
# 3. REVIEW MISSING VALUES
# --------------------

print("\n==========================================")
print("MISSING VALUES BEFORE INSERTING")
print("==========================================")

print(df.isnull().sum())


# --------------------
# 4. INTENTIONALLY ADD 5 MISSING VALUES
#    TO MEAN RADIUS
# --------------------

np.random.seed(42)

# Choose 5 random records
random_rows = np.random.choice(df.index, size=5, replace=False)

# Replace mean radius with NaN
df.loc[random_rows, "mean radius"] = np.nan

print("\n==========================================")
print("MISSING VALUES AFTER ARTIFICIAL INSERTION")
print("==========================================")

print(df["mean radius"].isnull().sum())


# --------------------
# 5. PROCESS MISSING VALUES
# --------------------

# The median is used because it is less influenced by
# extreme values and outliers than the mean.

median_radius = df["mean radius"].median()

df["mean radius"] = df["mean radius"].fillna(median_radius)


print("\n==========================================")
print("MISSING VALUES AFTER HANDLING")
print("==========================================")

print(df["mean radius"].isnull().sum())

print("\nMedian used for missing mean radius:")
print(median_radius)


# --------------------
# 6. REVIEW CLASS DISTRIBUTION
# --------------------

print("\n==========================================")
print("CLASS DISTRIBUTION")
print("==========================================")

class_counts = df["target"].value_counts().sort_index()

print(class_counts)

print("\nClass names:")
print("0 =", data.target_names[0])
print("1 =", data.target_names[1])


# Display the percentage distribution
class_percentage = df["target"].value_counts(
    normalize=True
).sort_index() * 100

print("\nClass percentages:")
print(class_percentage)


# --------------------
# 7. CHECK FOR DATASET IMBALANCE
# --------------------

minority_count = class_counts.min()
majority_count = class_counts.max()

imbalance_ratio = minority_count / majority_count

print("\nImbalance ratio:", imbalance_ratio)

if imbalance_ratio < 0.8:
    print("The dataset is somewhat imbalanced.")
else:
    print("The dataset is relatively balanced.")


# --------------------
# 8. APPLY SMOTE OVERSAMPLING
# --------------------

# Split features from the target
X = df.drop("target", axis=1)
y = df["target"]


print("\n==========================================")
print("CLASS COUNTS BEFORE SMOTE")
print("==========================================")

print(y.value_counts().sort_index())


# Perform SMOTE sampling
smote = SMOTE(random_state=42)

X_smote, y_smote = smote.fit_resample(X, y)


print("\n==========================================")
print("CLASS COUNTS AFTER SMOTE")
print("==========================================")

print(pd.Series(y_smote).value_counts().sort_index())


# --------------------
# 9. FIND OUTLIERS WITH THE IQR METHOD
#    FEATURES: AVERAGE RADIUS AND AVERAGE TEXTURE
# --------------------

def detect_outliers_iqr(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = column[
        (column < lower_bound) |
        (column > upper_bound)
    ]

    return Q1, Q3, IQR, lower_bound, upper_bound, outliers


# Mean radius
Q1_radius, Q3_radius, IQR_radius, lower_radius, upper_radius, outliers_radius = \
    detect_outliers_iqr(df["mean radius"])


# Mean texture
Q1_texture, Q3_texture, IQR_texture, lower_texture, upper_texture, outliers_texture = \
    detect_outliers_iqr(df["mean texture"])


print("\n==========================================")
print("OUTLIER DETECTION")
print("==========================================")

print("\nMean Radius:")
print("Q1:", Q1_radius)
print("Q3:", Q3_radius)
print("IQR:", IQR_radius)
print("Lower Bound:", lower_radius)
print("Upper Bound:", upper_radius)
print("Number of Outliers:", len(outliers_radius))


print("\nMean Texture:")
print("Q1:", Q1_texture)
print("Q3:", Q3_texture)
print("IQR:", IQR_texture)
print("Lower Bound:", lower_texture)
print("Upper Bound:", upper_texture)
print("Number of Outliers:", len(outliers_texture))


# --------------------
# 10. DISPLAY OUTLIERS WITH A BOXPLOT
# --------------------

plt.figure(figsize=(10, 6))

plt.boxplot(
    [
        df["mean radius"],
        df["mean texture"]
    ],
    labels=["Mean Radius", "Mean Texture"]
)

plt.title("Outlier Detection using Boxplot")
plt.ylabel("Values")
plt.grid(True)

plt.show()


# --------------------
# 11. APPLY LOG TRANSFORMATION TO MEAN AREA
# --------------------

# Inspect the original distribution

plt.figure(figsize=(8, 5))

plt.hist(
    df["mean area"],
    bins=30
)

plt.title("Mean Area Before Log Transformation")
plt.xlabel("Mean Area")
plt.ylabel("Frequency")

plt.show()


# Perform the log transformation
df["log mean area"] = np.log1p(df["mean area"])


# --------------------
# 12. SHOW HISTOGRAM AFTER LOG TRANSFORMATION
# --------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["log mean area"],
    bins=30
)

plt.title("Mean Area After Log Transformation")
plt.xlabel("Log(Mean Area + 1)")
plt.ylabel("Frequency")

plt.show()


# --------------------
# 13. GET FEATURES READY FOR MACHINE LEARNING
# --------------------

# Remove the target column
X = df.drop("target", axis=1)

y = df["target"]


# --------------------
# 14. SPLIT INTO TRAINING AND TEST SETS
# --------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\n==========================================")
print("TRAIN TEST SPLIT")
print("==========================================")

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# --------------------
# 15. SCALE THE FEATURES
# --------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


print("\n==========================================")
print("SCALING COMPLETE")
print("==========================================")

print("Scaled training shape:", X_train_scaled.shape)
print("Scaled testing shape:", X_test_scaled.shape)


# --------------------
# WHY DO WE SCALE?
# --------------------

print("\nWhy is scaling important?")
print(
    "StandardScaler puts features on a similar scale. "
    "This is especially important for distance-based models "
    "such as KNN and SVM because large-valued features could "
    "otherwise dominate the distance calculations."
)


# --------------------
# 16. PCA - REDUCE TO TWO COMPONENTS
# --------------------

pca = PCA(n_components=2)

X_pca = pca.fit_transform(
    scaler.fit_transform(X)
)


print("\n==========================================")
print("PCA RESULTS")
print("==========================================")

print("Original number of features:", X.shape[1])

print("Number of PCA components:", X_pca.shape[1])

print("Explained variance ratio:")
print(pca.explained_variance_ratio_)

print(
    "\nTotal variance explained:",
    pca.explained_variance_ratio_.sum()
)


# --------------------
# 17. PLOT PCA COMPONENTS
# --------------------

plt.figure(figsize=(9, 6))

plt.scatter(
    X_pca[y == 0, 0],
    X_pca[y == 0, 1],
    label="Malignant"
)

plt.scatter(
    X_pca[y == 1, 0],
    X_pca[y == 1, 1],
    label="Benign"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title("PCA: Breast Cancer Dataset")

plt.legend()
plt.grid(True)

plt.show()


# --------------------
# 18. TRAIN AN SVM MODEL
# --------------------

svm_model = SVC(
    kernel="rbf",
    random_state=42
)

svm_model.fit(
    X_train_scaled,
    y_train
)

svm_predictions = svm_model.predict(
    X_test_scaled
)

svm_accuracy = accuracy_score(
    y_test,
    svm_predictions
)


# --------------------
# 19. TRAIN A KNN MODEL
# --------------------

knn_model = KNeighborsClassifier(
    n_neighbors=5
)

knn_model.fit(
    X_train_scaled,
    y_train
)

knn_predictions = knn_model.predict(
    X_test_scaled
)

knn_accuracy = accuracy_score(
    y_test,
    knn_predictions
)


# --------------------
# 20. COMPARE MODEL ACCURACY
# --------------------

print("\n==========================================")
print("CLASSIFIER ACCURACY")
print("==========================================")

print("SVM Accuracy:", svm_accuracy)

print("KNN Accuracy:", knn_accuracy)


print("\nAccuracy Comparison:")

if svm_accuracy > knn_accuracy:
    print("SVM has higher accuracy on this train-test split.")
elif knn_accuracy > svm_accuracy:
    print("KNN has higher accuracy on this train-test split.")
else:
    print("Both models have the same accuracy.")


# --------------------
# 21. FINAL RESULTS AND OBSERVATION
# --------------------

print("\n==========================================")
print("FINAL OBSERVATION")
print("==========================================")

print(
    "Scaling is particularly important for this dataset because "
    "the features have very different numerical ranges and both "
    "KNN and SVM are sensitive to feature scale."
)

print(
    "Outlier detection and imbalance handling are also useful, "
    "but scaling can have a direct effect on distance-based "
    "models such as KNN and SVM."
)