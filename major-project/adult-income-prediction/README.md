# 💰 Adult Income Prediction

> **Can we predict whether someone earns more or less than $50,000 a year?**
> This project answers that question using real census data from 48,000+ people — and the answer turns out to hinge on just two things.

---

## 🧩 The Problem

Not everyone fills out a salary field on a form. But census data captures dozens of signals — age, education, occupation, investment activity — that together paint a surprisingly clear picture of where someone lands on the income scale.

Using a dataset of real individuals, we built a model that classifies people into two groups: **≤$50K** or **>$50K** per year, and then asked: *what actually drives that prediction?*

---

## 💡 The Short Answer

Two features dominate everything else:

- 📈 **Capital gains** — high earners had **6× more** investment activity than low earners
- 🎓 **Education level** — high earners averaged **~2 more years** of schooling

Both findings reflect how income actually works in the real world — which makes the model not just accurate, but *explainable*.

---

## 📊 Key Findings at a Glance

| Finding | Detail |
|---|---|
| 🥇 **Strongest predictor** | Capital gain — high earners had 6× more investment activity |
| 🥈 **Second strongest** | Education level — high earners averaged ~2 more years |
| ⚖️ **Class imbalance** | 76% earn ≤$50K vs. 24% earn >$50K (3.2:1 ratio) |
| ❓ **Missing data** | ~5.7% of rows had unknown `workclass` and `occupation` simultaneously |
| 🏆 **Best model** | Random Forest outperformed Logistic Regression |

---

## 📁 Dataset

**Source:** UCI Adult Income (Census) Dataset
**File:** `adult-income.csv`
**Rows:** 48,842 (48,790 after removing 52 duplicates)
**Target:** `income` — binary label (`<=50K` or `>50K`)

| Column | Type | Description |
|---|---|---|
| `age` | Numeric | Age of the individual |
| `workclass` | Categorical | Employment type (Private, Gov, Self-emp, etc.) |
| `fnlwgt` | Numeric | Census sampling weight — **dropped** (not predictive) |
| `education` | Categorical | Highest education level — **dropped** (redundant with `educational-num`) |
| `educational-num` | Numeric | Education encoded as an ordinal number |
| `marital-status` | Categorical | Marital status (7 categories) |
| `occupation` | Categorical | Job type (15 categories) |
| `relationship` | Categorical | Family role (Husband, Wife, Own-child, etc.) |
| `race` | Categorical | Race |
| `gender` | Categorical | Gender |
| `capital-gain` | Numeric | Investment gains |
| `capital-loss` | Numeric | Investment losses |
| `hours-per-week` | Numeric | Weekly working hours |
| `native-country` | Categorical | Country of origin (42 unique values) |
| `income` | Target | `<=50K` or `>50K` |

---

## ⚙️ How It Works (Simply Put)

1. **Explore the data** — Profile every column, catch hidden missing values, and document the class imbalance.
2. **Clean & engineer features** — Handle outliers, collapse sparse categories, and log-transform skewed columns.
3. **Build preprocessing pipelines** — Separate pipelines for distance-based vs. tree-based models, all leak-proof.
4. **Train two models** — Logistic Regression as a baseline, Random Forest as the main model.
5. **Rank the predictors** — Use permutation importance to find out what the model actually learned.

---

## 🔍 What the Data Looked Like Before Modeling

| Issue | Detail |
|---|---|
| 🔎 Hidden missing values | `?` found in `workclass`, `occupation`, `native-country` — converted to `NaN` |
| 📉 Outliers | `hours-per-week` (27.6%), `capital-gain` (8.3%), `capital-loss` (4.7%) |
| ⚖️ Target imbalance | ~3.2:1 ratio — addressed with class weighting |
| 🔁 Redundant features | `education` and `educational-num` encode the same information |
| 🗑️ Non-predictive feature | `fnlwgt` is a census artifact with no real-world income signal |

---

## 🛠️ Feature Engineering

- **Log-transformed** `capital-gain` and `capital-loss` using `log1p` to handle heavy zero-skew
- **Collapsed `native-country`** from 42 categories into a binary flag: `United-States` / `Non-US`
- **Dropped `fnlwgt`** — a census sampling weight, not a personal attribute
- **Dropped `education`** — fully redundant with `educational-num`

---

## 🤖 Models Trained

| Model | Notes |
|---|---|
| 📐 **Logistic Regression** | `max_iter=1000`, stratified 75/25 train-test split |
| 🌲 **Random Forest** | `n_jobs=-1` for parallel training, `random_state=43` |

Two separate preprocessing pipelines were used depending on model type:

**For distance-based models (Logistic Regression):**
Median imputation → StandardScaler for numerics; mode imputation → OneHotEncoder for categoricals.

**For tree-based models (Random Forest):**
Median imputation (no scaling needed); mode imputation → OrdinalEncoder for categoricals.

---

## 🏆 Top Predictors (Permutation Importance)

Permutation importance was computed on the Random Forest test set over 10 repeats:

| Rank | Feature | Mean Accuracy Drop | What It Signals |
|---|---|---|---|
| 🥇 1 | `capital-gain` | ~0.048 | Access to investment markets |
| 🥈 2 | `educational-num` | ~0.029 | Long-term earning potential |
| 🥉 3 | `relationship` | — | Household structure & dual incomes |
| 4 | `occupation` | — | Job type and sector |
| 5 | `hours-per-week` | — | Work intensity |

---

## 🛠️ Tools Used

- **Python** — the programming language
- **Pandas & NumPy** — for data manipulation
- **Matplotlib & Seaborn** — for charts and visualizations
- **Scikit-learn** — for preprocessing, pipelines, and modeling
- **imbalanced-learn** — for pipeline support with imbalanced datasets
- **my_utils** *(local)* — custom profiling and plotting utilities

---

## 📁 File Structure

```
📦 project
 ┣ 📓 adult-income-prediction.ipynb   ← the full analysis notebook
 ┣ 📄 adult-income.csv                ← census dataset
 ┗ 📄 README.md                       ← you are here
```

---

## 📌 Business Takeaway

The model reveals that income inequality in this dataset is driven by factors that are both measurable and interpretable:

- 📈 **Capital gains** reflect access to investment markets — a privilege concentrated in higher income brackets
- 🎓 **Education** reflects long-term earning potential — each additional year of schooling correlates with higher pay

These aren't surprises. But having a model that *quantifies and ranks* them gives decision-makers a data-backed foundation for policies around education access, tax structure, and income support programs.

*Built as part of an income classification exercise. The goal: turn raw census data into an explainable, trustworthy prediction model.* 🚀
