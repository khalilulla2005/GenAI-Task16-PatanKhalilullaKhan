"""
Assignment 16 - Task 1: Support Vector Machine (SVM)
Dataset : Iris (Kaggle: https://www.kaggle.com/datasets/uciml/iris)
Flow    : load -> train/test split -> SVC(linear) -> SVC(rbf) -> compare accuracy
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save plots to file (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

RANDOM_STATE = 42
os.makedirs("plots", exist_ok=True)

# ------------------------------------------------------------- 1. Dataset
iris = load_iris(as_frame=True)      # identical data to the Kaggle Iris CSV
X, y = iris.data, iris.target        # X = DataFrame of 4 numerical features

print("=" * 60)
print("TASK 1 : SUPPORT VECTOR MACHINE (SVM)")
print("=" * 60)
print("\nFirst 5 rows:")
print(X.head())
print("\nShape:", X.shape, "| Classes:", list(iris.target_names))
print("\nClass distribution (counts):")
print(y.value_counts().sort_index())

# ------------------------------------------------------------- 2. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# SVM is distance-based -> standardise features (fit on TRAIN only!)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ------------------------------------------------ 3-5. Train both kernels & compare
accuracies = {}
for kernel in ("linear", "rbf"):
    svm = SVC(kernel=kernel, random_state=RANDOM_STATE)
    svm.fit(X_train_s, y_train)
    y_pred = svm.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    accuracies[kernel] = acc
    print(f"\n--- SVC(kernel='{kernel}') ---")
    print(f"Test accuracy: {acc:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification report:\n",
          classification_report(y_test, y_pred, target_names=iris.target_names))

best = max(accuracies, key=accuracies.get)
print(f">>> RESULT: best kernel = '{best}' "
      f"(linear={accuracies['linear']:.4f} | rbf={accuracies['rbf']:.4f})")

plt.figure(figsize=(6, 4))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()),
            hue=list(accuracies.keys()), palette="viridis", legend=False)
plt.ylim(0, 1.05)
plt.xlabel("Kernel")
plt.ylabel("Test Accuracy")
plt.title("Task 1: SVM Kernel Comparison (Iris)")
for i, v in enumerate(accuracies.values()):
    plt.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("plots/task1_svm_kernels.png", dpi=150)
plt.close()
print("Saved plot -> plots/task1_svm_kernels.png")
