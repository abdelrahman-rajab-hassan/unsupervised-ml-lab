# Adult Income Classification
### Predicting whether a person earns more than $50K/year — with and without feature engineering

---

## The Problem

Income inequality is one of the most studied socioeconomic issues, but predicting *individual* earning potential from demographic and employment data has concrete real-world value — from policy targeting to financial product design.

This project uses the **Adult Income dataset** (1994 U.S. Census) to build a binary classifier that answers one question:

> **Can we predict whether a person earns more than $50,000 per year — and does feature engineering help?**

The project is split into two parts:
- **Part 1** — Build a baseline Random Forest with no feature engineering and identify what drives predictions
- **Part 2** — Apply PCA and Variance Threshold feature selection, then compare all models

---

## Dataset at a Glance

**48,842 rows | 15 columns | Binary classification target**

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Person's age |
| `workclass` | Categorical | Employment sector |
| `fnlwgt` | Numeric | Census sampling weight — **dropped** |
| `education` | Categorical | Highest level of education |
| `educational-num` | Numeric | Education encoded as an ordinal number (1–16) |
| `marital-status` | Categorical | Marital status |
| `occupation` | Categorical | Type of job |
| `relationship` | Categorical | Household role (Husband, Wife, Own-child, etc.) |
| `race` | Categorical | Race |
| `gender` | Categorical | Gender |
| `capital-gain` | Numeric | Investment gains |
| `capital-loss` | Numeric | Investment losses |
| `hours-per-week` | Numeric | Average weekly working hours |
| `native-country` | Categorical | Country of origin |
| `income` | **Target** | `<=50K` or `>50K` |

<!-- IMAGE 1 -->
> **Insert:** Screenshot of `df.head()` — shows the raw data structure with all 15 columns

---

## Part 1 — Baseline Model (No Feature Engineering)

---

### Step 1 — Exploratory Data Analysis

We profiled every column before touching any model — checking distributions, value counts, and data quality.

**Data quality checks:**

| Check | Result |
|---|---|
| Null values | None — dataset is complete |
| Duplicate rows | **52 found and removed** |
| Hidden missing values | `?` placeholder found in `workclass`, `occupation`, and `native-country` |

We then plotted univariate distributions for every column to understand the shape of the data.

<!-- IMAGE 2 -->
> **Insert:** The `income` target distribution plot — shows the **class imbalance** (~76% earn `<=50K`, ~24% earn `>50K`)

<!-- IMAGE 3 -->
> **Insert:** 2–3 EDA univariate plots (e.g., `age` histogram, `education` bar chart, `occupation` bar chart) — shows the variety in how features are distributed

---

### Step 2 — Preprocessing

We split the data using a **stratified 75/25 train-test split** to preserve the class ratio, then built a `ColumnTransformer` pipeline:

```
Numeric columns   → SimpleImputer (median strategy)
Categorical columns → SimpleImputer (most_frequent) → OrdinalEncoder
```

`fnlwgt` was dropped — it is a census sampling weight with no predictive meaning for individual income.

---

### Step 3 — Train the Baseline Random Forest

A **Random Forest Classifier** with default hyperparameters (`random_state=43`) was trained directly on the preprocessed features — no dimensionality reduction, no engineered features.

**Baseline performance:**

| Metric | Score |
|---|---|
| Accuracy | **0.86** |
| Precision (`>50K`) | **0.76** |
| Recall (`>50K`) | **0.63** |
| F1-score (`>50K`) | **0.69** |

---

### Step 4 — Permutation Feature Importance

We used **permutation importance** (10 repeats on the test set) to rank which features most affected the model's accuracy when shuffled.

<!-- IMAGE 4 -->
> **Insert:** Horizontal bar chart — Top 10 features by permutation importance (Baseline Random Forest)

**Why the top features make sense:**

| Feature | Explanation |
|---|---|
| `capital-gain` | Investment income is strongly associated with higher overall wealth — by far the #1 signal |
| `relationship` | Household role (e.g. "Husband") correlates with higher reported income in this dataset |
| `marital-status` | Married individuals tend to have higher income — overlaps with relationship |
| `educational-num` | Education directly reflects earning potential — every extra level raises income probability |
| `occupation` | Professional and managerial roles command higher salaries |
| `age` | Income generally rises with career experience through mid-life |

---

### Step 5 — Explanatory Visualizations

We selected **`age`** and **`educational-num`** from the top 10 to visualize their relationship with the target — chosen because they are universally understood and tell the clearest story to a non-technical audience.

<!-- IMAGE 5 -->
> **Insert:** `age` vs `income` plot
>
> **Insight:** People in their **40s and 50s** are most likely to earn over $50K, with the proportion peaking around ages 45–55. Younger workers (under 30) and older retirees (over 65) are far less likely to fall in the higher bracket — confirming the classic mid-career earnings curve.

<!-- IMAGE 6 -->
> **Insert:** `educational-num` vs `income` plot
>
> **Insight:** Below education level 12 (high school diploma), almost nobody earns over $50K. Starting at level 13 (Bachelor's degree), the proportion earning over $50K jumps sharply — reaching nearly **75% at the highest levels** (Master's and Doctorate). Education is one of the most consistent and explainable predictors in the dataset.

---

## Part 2 — Feature Engineering

---

### Step 6 — Apply PCA

After the baseline, we tested whether adding **principal components** as new features could capture latent structure the model was missing.

**Pipeline:**
```
Numeric  → SimpleImputer (median) → StandardScaler
Categorical → SimpleImputer (most_frequent) → OrdinalEncoder
→ PCA (3 components)
→ Concatenate: original 13 features + 3 PCs = 16 features total
```

We fitted PCA on the training data only, then transformed both train and test to avoid leakage.

**Result:**

| Metric | Baseline | PCA + Original |
|---|---|---|
| Accuracy | **0.86** | 0.86 |
| Precision (`>50K`) | **0.76** | 0.74 |
| Recall (`>50K`) | **0.63** | 0.62 |
| F1-score (`>50K`) | **0.69** | 0.68 |

Adding PCA components **did not improve** the model. The principal components captured compressed, blended information that the Random Forest was already extracting more precisely from the original features directly.

---

### Step 7 — Apply Variance Threshold Feature Selection

We applied `VarianceThreshold(threshold=0.1)` to the 16-feature set (original 13 + 3 PCs) to filter out any near-zero-variance features.

```
Features before filtering: 16
Features after filtering:   16
```

All 16 features exceeded the variance threshold — none were redundant enough to remove. This confirms the dataset's features each carry meaningful signal.

---

### Step 8 — Final Model Comparison

<!-- IMAGE 7 -->
> **Insert:** The model comparison table as a styled image, or keep the markdown table below

| Metric | Baseline | PCA + Original | After Feature Selection |
|---|---|---|---|
| Accuracy | **0.86** | 0.86 | 0.86 |
| Precision (`>50K`) | **0.76** | 0.74 | 0.74 |
| Recall (`>50K`) | **0.63** | 0.62 | 0.62 |
| F1-score (`>50K`) | **0.69** | 0.68 | 0.68 |

<!-- IMAGE 8 -->
> **Insert:** Horizontal bar chart — Top 10 features by permutation importance (Feature Selection model)
>
> **Notable shift:** `age` jumped from #4 to #1, while `capital-gain` dropped from #1 to #3, and `relationship` fell from #2 to #10. Despite these rank changes, performance did not improve.

---

## Conclusions

**The baseline model is the best performer across all metrics.** Feature engineering via PCA and Variance Threshold did not improve — and marginally hurt — the model.

| Question | Answer |
|---|---|
| Best model | Baseline Random Forest (no engineering) |
| Most important feature (baseline) | `capital-gain` |
| Most important feature (after PCA) | `age` |
| Did PCA help? | No — introduced redundant compressed information |
| Did Variance Threshold help? | No — removed zero features, identical result |
| Core takeaway | Random Forests already learn the best splits from raw features; compressing them via PCA removes the fine-grained signal trees rely on |

---

## Why Feature Engineering Failed Here

This is an important result, not a failure of process. Random Forests are **inherently non-linear** and **feature-selective** — they build hundreds of trees, each sampling random subsets of features and finding the most informative splits. Adding PCA components gives the model blended, rotated versions of features it already has direct access to. The trees gain nothing and lose resolution.

Variance Threshold confirmed there were no low-signal features to discard — every feature carries enough variance to be useful. In a dataset this clean and well-structured, the best move is to trust the baseline.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `pandas` / `numpy` | Data manipulation |
| `seaborn` / `matplotlib` | Visualization |
| `scikit-learn` | Modeling, preprocessing, evaluation, PCA, feature selection |
| `imbalanced-learn` | Pipeline compatibility |
| `my_utils` (local) | Custom profiling and plotting utilities |

---

## How to Run

1. Place `adult-income.csv` in the same directory as the notebook.
2. Ensure `my_utils.py` is one level up (`../my_utils.py`).
3. Install dependencies:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn imbalanced-learn
```

4. Open `adult-income copy.ipynb` and run all cells top to bottom.

---

## Project Structure

```
adult-income/
├── adult-income.csv              # Raw dataset (UCI Adult Income)
├── adult-income copy.ipynb       # Full analysis notebook (Parts 1 & 2)
└── README.md                     # This file
```
