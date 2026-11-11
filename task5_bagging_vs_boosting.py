"""
Assignment 16 - Task 5: Bagging vs Boosting (Conceptual + Practical)

BAGGING (Bootstrap AGGregatING)
  * Train MANY models IN PARALLEL, each on a random bootstrap sample
    (sampling with replacement) of the training data.
  * Final answer = majority vote / average of all models.
  * Main effect: reduces VARIANCE (stability). Example: Random Forest.

BOOSTING
  * Train models SEQUENTIALLY; each new model focuses on the mistakes
    (misclassified samples get more weight) of the previous ones.
  * Final answer = weighted vote of all models.
  * Main effect: reduces BIAS (weak learners -> strong learner). Example: AdaBoost.

Practical part: train BaggingClassifier and AdaBoostClassifier, compare.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
os.makedirs("plots", exist_ok=True)

print("=" * 60)
print("TASK 5 : BAGGING vs BOOSTING")
print("=" * 60)
print("\nConcept:")
print("  Bagging  -> parallel  models on bootstrap samples -> majority vote (cuts variance)")
print("  Boosting -> sequential models fixing past mistakes  -> weighted vote (cuts bias)")

iris = load_iris(as_frame=True)   # Kaggle: https://www.kaggle.com/datasets/uciml/iris
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# ------------------------------------------------- Train both ensembles
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
    n_estimators=50, random_state=RANDOM_STATE
)
boosting = AdaBoostClassifier(n_estimators=50, random_state=RANDOM_STATE)

accuracies = {}
for name, model in (("BaggingClassifier", bagging), ("AdaBoostClassifier", boosting)):
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    accuracies[name] = acc
    print(f"\n{name}: test accuracy = {acc:.4f}")

best = max(accuracies, key=accuracies.get)
print(f"\n>>> RESULT: better performer on this dataset = {best}")

plt.figure(figsize=(6, 4))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()),
            hue=list(accuracies.keys()), palette="magma", legend=False)
plt.ylim(0, 1.05)
plt.ylabel("Test Accuracy")
plt.title("Task 5: Bagging vs Boosting (AdaBoost)")
for i, v in enumerate(accuracies.values()):
    plt.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("plots/task5_bagging_vs_boosting.png", dpi=150)
plt.close()
print("Saved plot -> plots/task5_bagging_vs_boosting.png")
