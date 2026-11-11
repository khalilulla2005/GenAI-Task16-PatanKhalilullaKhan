"""
Assignment 16 - Task 3: Train vs Validation vs Test Split
1. Split data into Training (60%) / Validation (20%) / Test (20%)
2. Train on the TRAINING set
3. Tune one hyper-parameter (max_depth) using the VALIDATION set
4. Evaluate the final model once on the TEST set

Why three sets?  The validation set is used to choose hyper-parameters.
If we tuned parameters on the test set we would 'leak' information and the
test score would no longer be an honest estimate of real-world performance.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
os.makedirs("plots", exist_ok=True)

print("=" * 60)
print("TASK 3 : TRAIN / VALIDATION / TEST SPLIT")
print("=" * 60)

iris = load_iris(as_frame=True)   # Kaggle: https://www.kaggle.com/datasets/uciml/iris
X, y = iris.data, iris.target

# --------------------------------------------- 1. 60 / 20 / 20 stratified split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=RANDOM_STATE, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
)
print(f"Train size : {len(X_train)}  (60%)")
print(f"Valid size : {len(X_val)}  (20%)")
print(f"Test  size : {len(X_test)}  (20%)")

# ----------------------------- 2-3. Train on train set, tune max_depth on val set
rows = []
for depth in range(1, 11):
    model = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)                      # learns from TRAIN only
    rows.append({
        "max_depth": depth,
        "train_accuracy":      round(model.score(X_train, y_train), 4),
        "validation_accuracy": round(model.score(X_val,   y_val),   4),
    })
table = pd.DataFrame(rows)
print("\nValidation tuning table:")
print(table.to_string(index=False))

best_depth = int(table.loc[table["validation_accuracy"].idxmax(), "max_depth"])
print(f"\n>>> Best max_depth chosen by VALIDATION set: {best_depth}")

# --------------------------------------------- 4. Final honest test evaluation
final_model = DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_STATE)
final_model.fit(X_train, y_train)
final_test_acc = final_model.score(X_test, y_test)
print(f">>> FINAL MODEL (max_depth={best_depth})  TEST accuracy: {final_test_acc:.4f}")

plt.figure(figsize=(7, 4.5))
plt.plot(table["max_depth"], table["train_accuracy"],      marker="o", label="Train")
plt.plot(table["max_depth"], table["validation_accuracy"], marker="s", label="Validation")
plt.axvline(best_depth, color="green", linestyle="--", alpha=0.7,
            label=f"chosen depth = {best_depth}")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Task 3: Hyper-parameter tuning with a validation set")
plt.legend()
plt.tight_layout()
plt.savefig("plots/task3_validation_curve.png", dpi=150)
plt.close()
print("Saved plot -> plots/task3_validation_curve.png")
