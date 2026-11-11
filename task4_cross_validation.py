"""
Assignment 16 - Task 4: Cross-Validation (K-Fold, k=5)
1. Apply K-Fold cross validation with k=5
2. Compute the average accuracy
3. Compare single train-test accuracy vs cross-validation accuracy

Why?  A single split accuracy depends on luck (which rows ended up in test).
K-Fold trains/tests on 5 different splits and averages -> more reliable estimate.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
os.makedirs("plots", exist_ok=True)

print("=" * 60)
print("TASK 4 : K-FOLD CROSS-VALIDATION (k=5)")
print("=" * 60)

iris = load_iris(as_frame=True)   # Kaggle: https://www.kaggle.com/datasets/uciml/iris
X, y = iris.data, iris.target

model = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)

# ------------------------------------------------- Single train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
model.fit(X_train, y_train)
single_acc = model.score(X_test, y_test)
print(f"\nSingle train-test split accuracy : {single_acc:.4f}")

# ------------------------------------------------- 5-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy")
print("\nFold accuracies:", np.round(cv_scores, 4))
print(f"CV mean accuracy                 : {cv_scores.mean():.4f}")
print(f"CV std  deviation                : {cv_scores.std():.4f}")

# ------------------------------------------------- Comparison
print(f"\n>>> COMPARISON: single-split={single_acc:.4f}  vs  5-fold-CV={cv_scores.mean():.4f}")
print("    CV gives a more stable estimate because every sample is used")
print("    in a test fold exactly once -> no 'lucky/unlucky' split effect.")

plt.figure(figsize=(6, 4))
bars = plt.bar(["Single\ntrain-test", "5-Fold CV\n(mean)"],
               [single_acc, cv_scores.mean()],
               color=["#4C72B0", "#55A868"], width=0.5)
plt.errorbar([1], [cv_scores.mean()], yerr=[cv_scores.std()],
             fmt="none", ecolor="black", capsize=8, linewidth=2)
for b, v in zip(bars, [single_acc, cv_scores.mean()]):
    plt.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
             ha="center", fontweight="bold")
plt.ylim(0, 1.1)
plt.ylabel("Accuracy")
plt.title("Task 4: Single split vs 5-Fold Cross-Validation")
plt.tight_layout()
plt.savefig("plots/task4_cv_comparison.png", dpi=150)
plt.close()
print("Saved plot -> plots/task4_cv_comparison.png")
