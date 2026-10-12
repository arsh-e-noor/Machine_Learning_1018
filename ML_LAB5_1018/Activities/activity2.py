# ============================================================================
#  Name             : ARSHENOOR
#  Registration No. : 23-NTU-CS-1018
#  Section          : BSCS 7th A
#  Course           : Machine Learning Lab
#  Lab              : Lab 05  -  Activity 02
#  Task             : SVM for Brain Tumor Detection from MRI images
#  Dataset          : brain_mri/  (folders "yes" and "no" of MRI images)
# ============================================================================

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "Dataset", "brain_mri")
PLOT_DIR = os.path.join(BASE_DIR, "Plots")
os.makedirs(PLOT_DIR, exist_ok=True)

IMG_SIZE = 64  # images are resized to 64 x 64 as the activity asks

# ---------------------------------------------------------------------------
# STEP 1 - Load and preprocess all MRI images
#          grayscale  ->  resize to 64x64  ->  flatten
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1 : LOADING AND PREPROCESSING THE IMAGES")
print("=" * 70)

X = []
y = []

# folder "yes" = tumour present (1), folder "no" = no tumour (0)
for folder_name, label in [("no", 0), ("yes", 1)]:
    folder = os.path.join(DATA_DIR, folder_name)

    if not os.path.isdir(folder):
        raise SystemExit(
            "\nFolder not found: {}\n"
            "Place the dataset so it looks like this:\n"
            "    Dataset/brain_mri/yes/   (MRI images WITH a tumour)\n"
            "    Dataset/brain_mri/no/    (MRI images WITHOUT a tumour)\n".format(folder)
        )

    count = 0
    for file_name in sorted(os.listdir(folder)):
        if not file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif")):
            continue
        try:
            img = Image.open(os.path.join(folder, file_name))
            img = img.convert("L")                       # convert to grayscale
            img = img.resize((IMG_SIZE, IMG_SIZE))       # resize to 64 x 64
            X.append(np.asarray(img, dtype=np.float32).flatten())   # flatten
            y.append(label)
            count += 1
        except Exception as err:
            print("  skipped {} ({})".format(file_name, err))

    print("  {:>3} images loaded from '{}'".format(count, folder_name))

X = np.array(X)
y = np.array(y)

print("\nFeature matrix shape :", X.shape, " (each row is one flattened image)")
print("Labels               : {} without tumour, {} with tumour".format(
    int((y == 0).sum()), int((y == 1).sum())))

# ---------------------------------------------------------------------------
# STEP 2 - Split the data into training and testing sets (80:20)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining images :", X_train.shape[0])
print("Testing images  :", X_test.shape[0])

# ---------------------------------------------------------------------------
# STEP 3 - Normalize the features using StandardScaler
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeatures normalized with StandardScaler.")

# ---------------------------------------------------------------------------
# STEP 4 - Train an SVM with the RBF kernel, C = 10, gamma = 'scale'
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4 : TRAINING THE SVM  (kernel = rbf, C = 10, gamma = 'scale')")
print("=" * 70)

svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
svm.fit(X_train_scaled, y_train)

print("\nTraining finished. Support vectors used:", svm.n_support_.sum())

# ---------------------------------------------------------------------------
# STEP 5 - Evaluate the model
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5 : EVALUATION")
print("=" * 70)

y_pred = svm.predict(X_test_scaled)

print("\nAccuracy : {:.4f}".format(accuracy_score(y_test, y_pred)))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(
    "\nClassification Report:\n",
    classification_report(y_test, y_pred, target_names=["No Tumour", "Tumour"], digits=4),
)

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["No Tumour", "Tumour"]
).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
ax.set_title("Activity 2 - Confusion Matrix (SVM, RBF kernel)")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "activity2_confusion_matrix.png"), dpi=120)
plt.show()

# ---------------------------------------------------------------------------
# STEP 6 - Display a few test images with their predicted and actual labels
# ---------------------------------------------------------------------------
names = {0: "No Tumour", 1: "Tumour"}
n_show = min(8, len(X_test))

fig, axes = plt.subplots(2, 4, figsize=(13, 8))
for ax, i in zip(axes.ravel(), range(n_show)):
    ax.imshow(X_test[i].reshape(IMG_SIZE, IMG_SIZE), cmap="gray")
    correct = y_pred[i] == y_test[i]
    ax.set_title(
        "Predicted: {}\nActual: {}".format(names[y_pred[i]], names[y_test[i]]),
        fontsize=9,
        color="green" if correct else "red",
    )
    ax.axis("off")

for ax in axes.ravel()[n_show:]:
    ax.axis("off")

fig.suptitle("Activity 2 - Sample Test Images (green = correct, red = wrong)")
plt.tight_layout()
plt.subplots_adjust(top=0.86, hspace=0.28)
plt.savefig(os.path.join(PLOT_DIR, "activity2_sample_predictions.png"), dpi=120)
plt.show()

print("\nPlots saved in the Plots folder.")
print("Activity 02 completed successfully.")
