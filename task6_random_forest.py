"""
Assignment 16 - Task 6: Random Forest (Bagging)
1. Train a RandomForestClassifier
2. Compare with a single Decision Tree and a Bagging Classifier
3. Print feature importance

Random Forest = Bagging of decision trees + random feature subset at each
split -> trees are less correlated, so the ensemble generalises better.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
os.makedirs("plots", exist_ok=True)

print("=" * 60)
print("TASK 6 : RANDOM FOREST (an example of Bagging)")
print("=" * 60)

iris = load_iris(as_frame=True)   # Kaggle: https://www.kaggle.com/datasets/uciml/iris
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# ------------------------------------------------- 1-2. Train the 3 models
models = {
    "Single Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Bagging Classifier":   BaggingClassifier(
                                estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
                                n_estimators=50, random_state=RANDOM_STATE),
    "Random Forest":        RandomForestClassifier(
                                n_estimators=100, random_state=RANDOM_STATE),
}

accuracies = {}
print("\nModel comparison (test accuracy):")
for name, model in models.items():
    model.fit(X_train, y_train)
    accuracies[name] = accuracy_score(y_test, model.predict(X_test))
    print(f"  {name:<22}: {accuracies[name]:.4f}")

best = max(accuracies, key=accuracies.get)
print(f"\n>>> RESULT: best model on this 30-sample test split = {best}")

# A 30-sample test split is noisy -> also report 5-fold CV (robust estimate)
print("\nRobust comparison using 5-fold cross-validation (mean +/- std):")
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = {}
for name, model in models.items():
    cv = cross_val_score(model, X, y, cv=kf, scoring="accuracy")  # clones model
    cv_results[name] = cv
    print(f"  {name:<22}: {cv.mean():.4f} (+/- {cv.std():.4f})")
cv_best = max(cv_results, key=lambda k: cv_results[k].mean())
print(f"\n>>> RESULT: best model by cross-validation = {cv_best}")
print("    A single tree has high variance; bagging/forest average many")
print("    trees -> more stable and usually better on unseen data.")

plt.figure(figsize=(7, 4))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()),
            hue=list(accuracies.keys()), palette="crest", legend=False)
plt.ylim(0, 1.05)
plt.ylabel("Test Accuracy")
plt.title("Task 6: Single Tree vs Bagging vs Random Forest")
plt.xticks(rotation=12)
for i, v in enumerate(accuracies.values()):
    plt.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("plots/task6_model_comparison.png", dpi=150)
plt.close()
print("Saved plot -> plots/task6_model_comparison.png")

# ------------------------------------------------- 3. Feature importance
importances = pd.Series(models["Random Forest"].feature_importances_,
                        index=iris.feature_names).sort_values()
print("\nRandom Forest feature importance (most -> least important):")
for feat, imp in importances.sort_values(ascending=False).items():
    print(f"  {feat:<20}: {imp:.4f}")

plt.figure(figsize=(7, 4))
importances.sort_values().plot(kind="barh", color="#4C72B0")
plt.xlabel("Importance")
plt.title("Task 6: Random Forest Feature Importance (Iris)")
plt.tight_layout()
plt.savefig("plots/task6_feature_importance.png", dpi=150)
plt.close()
print("Saved plot -> plots/task6_feature_importance.png")
