# Assignment 16 — SVM, Trees, Ensembles, Validation & Unsupervised Learning

**Submitted by:** PATAN KHALILULLA KHAN
**Topic:** Advanced supervised algorithms, model validation strategies and ensemble learning

---

## 📊 Dataset

| Item | Detail |
|---|---|
| **Name** | Iris Flower Dataset |
| **Kaggle link** | https://www.kaggle.com/datasets/uciml/iris |
| **Samples** | 150 rows |
| **Features** | 4 numerical features: sepal length, sepal width, petal length, petal width (cm) |
| **Target** | Classification — 3 classes: *setosa*, *versicolor*, *virginica* (50 each) |

> The identical dataset ships with scikit-learn (`sklearn.datasets.load_iris`), so the
> scripts run without downloading the CSV; it is the same data published on Kaggle/UCI.

**Libraries used (allowed only):** scikit-learn, pandas, numpy, matplotlib, seaborn. No deep learning.

---

## 📁 Folder Structure

```
GenAI-Task16-PatanKhalilullaKhan/
├── README.md
├── requirements.txt
├── task1_svm.py                      # SVM linear vs RBF kernel
├── task2_decision_tree.py            # tree visualisation + underfit vs overfit
├── task3_train_validation_test.py    # 60/20/20 split + validation tuning
├── task4_cross_validation.py         # K-Fold CV (k=5) vs single split
├── task5_bagging_vs_boosting.py      # BaggingClassifier vs AdaBoostClassifier
├── task6_random_forest.py            # RandomForest + feature importance
├── plots/                            # 8 saved figures (evidence for every task)
└── outputs/                          # full console output of every script
```

## ▶️ How to Run

```bash
pip install -r requirements.txt
python task1_svm.py
python task2_decision_tree.py
python task3_train_validation_test.py
python task4_cross_validation.py
python task5_bagging_vs_boosting.py
python task6_random_forest.py
```

All plots are saved automatically inside `plots/`. Random seeds are fixed
(`random_state=42`) so results are 100% reproducible.

---

## ✅ Results Summary

### PART 1 — Advanced Supervised Algorithms

**Task 1 — SVM (`SVC`, features standardised, 80/20 stratified split)**

| Kernel | Test Accuracy |
|---|---|
| Linear | **1.0000** ✅ best |
| RBF | 0.9667 |

Iris is (almost) linearly separable, so the linear kernel wins slightly here.

**Task 2 — Decision Tree (underfit vs overfit)**

| max_depth | Train acc | Test acc |
|---|---|---|
| 1 | 0.6667 | 0.6667 → **underfitting** |
| 2 | 0.9667 | 0.9333 |
| 3 | 0.9833 | 0.9667 → **sweet spot** |
| 5 | 1.0000 | 0.9333 |
| 8 | 1.0000 | 0.9333 |
| full (None) | 1.0000 | 0.9333 → **overfitting** |

Tree structure visualised at depth 3 → `plots/task2_tree_structure.png`.

### PART 2 — Model Validation

**Task 3 — Train (60) / Validation (20) / Test (20) split**

- Tuned `max_depth` (1–10) on the **validation** set → best = **3**
- Final model evaluated once on **test** set → accuracy **1.0000**
- Validation set prevents "test-set peeking" while choosing hyper-parameters.

**Task 4 — K-Fold Cross Validation (k = 5)**

| Method | Accuracy |
|---|---|
| Single train-test split | 0.9667 |
| 5-Fold CV (mean ± std) | **0.9533 ± 0.034** |

Fold scores: `[1.000, 0.967, 0.933, 0.900, 0.967]` — CV is the more reliable estimate
because every sample is tested exactly once, removing the lucky/unlucky-split effect.

### PART 3 — Ensemble Learning

**Task 5 — Bagging vs Boosting**

- **Bagging:** many models trained *in parallel* on bootstrap samples → majority vote → reduces **variance**.
- **Boosting:** models trained *sequentially*, each fixing the previous model's mistakes (re-weighting) → reduces **bias**.

| Model | Test Accuracy |
|---|---|
| BaggingClassifier (50 trees) | **0.9667** ✅ |
| AdaBoostClassifier (50 stumps) | 0.9333 |

Iris is small & clean, so parallel bagging has the edge; boosting shines more on
harder, high-bias problems.

**Task 6 — Random Forest (Bagging)**

| Model | Test acc (30 samples) | 5-Fold CV mean |
|---|---|---|
| Single Decision Tree | 0.9333 | 0.9533 |
| Bagging Classifier | **0.9667** | 0.9533 |
| Random Forest (100 trees) | 0.9000 | **0.9600** ✅ |

Note: on a single 30-sample split, scores fluctuate; the robust **CV** view shows the
Random Forest on top, matching the theory (bagging + random feature subsets).

**Feature importance (Random Forest):**

| Feature | Importance |
|---|---|
| petal width (cm) | 0.437 |
| petal length (cm) | 0.432 |
| sepal length (cm) | 0.116 |
| sepal width (cm) | 0.015 |

→ Petal measurements are by far the most informative for Iris classification.

---

## 📈 Plots (in `plots/`)

1. `task1_svm_kernels.png` — kernel accuracy comparison
2. `task2_tree_structure.png` — decision tree diagram (depth 3)
3. `task2_underfit_overfit.png` — train vs test accuracy across depths
4. `task3_validation_curve.png` — validation-based tuning of max_depth
5. `task4_cv_comparison.png` — single split vs 5-fold CV
6. `task5_bagging_vs_boosting.png` — bagging vs AdaBoost
7. `task6_model_comparison.png` — tree vs bagging vs forest
8. `task6_feature_importance.png` — Random Forest feature importance

## 🔑 Key Takeaways

1. Feature scaling matters for distance-based models (SVM).
2. Trees need depth control: too shallow underfits, full depth memorises (overfits).
3. Tune hyper-parameters on a **validation** set (or CV), never on the test set.
4. Ensembles average out single-model mistakes: bagging cuts variance, boosting cuts bias.
5. Feature importance explains *what* the model actually learned.
