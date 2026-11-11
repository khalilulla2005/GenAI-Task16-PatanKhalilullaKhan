"""
Assignment 16 - Task 2: Decision Tree Algorithm
1. Train a DecisionTreeClassifier
2. Visualise the tree structure (depth limited)
3. Low max_depth  -> underfitting
   High max_depth -> overfitting
4. Compare train vs test accuracy
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

RANDOM_STATE = 42
os.makedirs("plots", exist_ok=True)

print("=" * 60)
print("TASK 2 : DECISION TREE (underfit vs overfit)")
print("=" * 60)

iris = load_iris(as_frame=True)   # Kaggle: https://www.kaggle.com/datasets/uciml/iris
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# ------------------------------------------------- 2. Visualise tree (depth=3)
viz_tree = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
viz_tree.fit(X_train, y_train)

plt.figure(figsize=(16, 7))
plot_tree(viz_tree,
          feature_names=iris.feature_names,
          class_names=list(iris.target_names),
          filled=True, rounded=True, fontsize=8)
plt.title("Task 2: Decision Tree structure (max_depth=3)")
plt.tight_layout()
plt.savefig("plots/task2_tree_structure.png", dpi=150)
plt.close()
print("Saved plot -> plots/task2_tree_structure.png")

# ------------------------------------ 3-4. Sweep depths: underfit -> overfit
depths = [1, 2, 3, 5, 8, None]           # None = grow until pure (deepest)
rows = []
for d in depths:
    tree = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE)
    tree.fit(X_train, y_train)
    rows.append({
        "max_depth":     "full (None)" if d is None else str(d),
        "train_accuracy": round(tree.score(X_train, y_train), 4),
        "test_accuracy":  round(tree.score(X_test,  y_test),  4),
    })
table = pd.DataFrame(rows)
print("\nTrain vs Test accuracy for different max_depth values:")
print(table.to_string(index=False))

print("\nObservation:")
print(" * max_depth=1  -> UNDERFITTING : low train AND test accuracy (too simple)")
print(" * full tree    -> OVERFITTING  : train accuracy ~1.000 but test accuracy drops")

plt.figure(figsize=(7, 4.5))
plt.plot(table["max_depth"], table["train_accuracy"], marker="o", label="Train accuracy")
plt.plot(table["max_depth"], table["test_accuracy"],  marker="s", label="Test accuracy")
plt.annotate("underfit", xy=(0, table["test_accuracy"].iloc[0]), xytext=(0.1, 0.75),
             arrowprops=dict(arrowstyle="->", color="red"), color="red")
plt.annotate("overfit", xy=(5, table["test_accuracy"].iloc[-1]), xytext=(3.5, 0.75),
             arrowprops=dict(arrowstyle="->", color="darkred"), color="darkred")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.ylim(0.70, 1.03)
plt.title("Task 2: Underfitting vs Overfitting")
plt.legend()
plt.tight_layout()
plt.savefig("plots/task2_underfit_overfit.png", dpi=150)
plt.close()
print("Saved plot -> plots/task2_underfit_overfit.png")
