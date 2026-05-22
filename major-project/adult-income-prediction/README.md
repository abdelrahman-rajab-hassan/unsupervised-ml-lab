# Adult Income Prediction

## What This Project Does (Non-Technical)

This project answers a simple question: **can we predict whether a person earns more or less than $50,000 per year?**

Using real census data from 48,000+ people, the project examines factors like age, education, occupation, and financial activity to build a model that classifies individuals into two income groups — `<=50K` and `>50K`.

The key insight from the analysis: **two things matter most for predicting high income — whether someone has investment income (capital gains) and how many years of education they completed.** Both findings align closely with how income works in the real world, making the model's output trustworthy and explainable to non-technical stakeholders.

---

## Key Findings at a Glance

| Finding | Detail |
|---|---|
| **Strongest income predictor** | Capital gain — high earners had 6x more investment activity |
| **Second strongest predictor** | Education level — high earners averaged ~2 more years than low earners |
| **Class imbalance** | 76% earn ≤$50K vs 24% earn >$50K (3.2:1 ratio) |
| **Missing data** | ~5.7% of rows had unknown `workclass` and `occupation` simultaneously |
| **Best model** | Random Forest outperformed Logistic Regression |

---

## Dataset

**Source:** UCI Adult Income (Census) Dataset  
**File:** `adult-income.csv`  
**Rows:** 48,842 (48,790 after removing 52 duplicates)  
**Target:** `income` — binary label (`<=50K` or `>50K`)

### Features

| Column | Type | Description |
|---|---|---|
| `age` | Numeric | Age of the individual |
| `workclass` | Categorical | Employment type (Private, Gov, Self-emp, etc.) |
| `fnlwgt` | Numeric | Census sampling weight — **dropped** (not predictive) |
| `education` | Categorical | Highest education level — **dropped** (redundant with `educational-num`) |
| `educational-num` | Numeric | Education encoded as ordinal number |
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

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

- Checked for nulls, duplicates, and hidden missing values (`?` characters)
- Profiled every column with distribution plots
- Identified outliers: `hours-per-week` had **27.6%** outliers — most severe column
- Documented the class imbalance (76/24 split) and its implications

### 2. Dataset Assessment Summary

| Issue | Detail |
|---|---|
| Hidden missing values | `?` found in `workclass`, `occupation`, `native-country` — converted to `NaN` |
| Outliers | `hours-per-week` (27.6%), `capital-gain` (8.3%), `capital-loss` (4.7%) |
| Target imbalance | ~3.2:1 ratio — requires class weighting or resampling |
| Redundant features | `education` and `educational-num` encode the same information |
| Non-predictive feature | `fnlwgt` is a census artifact with no real-world income signal |

### 3. Feature Engineering

- **Log-transformed** `capital-gain` and `capital-loss` using `log1p` to handle the heavy zero-skew
- **Collapsed `native-country`** from 42 categories into binary: `United-States` / `Non-US`
- **Dropped `fnlwgt`** — census sampling weight, not a personal attribute
- **Dropped `education`** — fully redundant with `educational-num`

### 4. Preprocessing Pipelines

Two separate preprocessing strategies were built depending on the model type:

**For distance-based models (Logistic Regression):**
- Numerical: median imputation → StandardScaler
- Categorical: mode imputation → OneHotEncoder

**For tree-based models (Random Forest):**
- Numerical: median imputation (no scaling needed)
- Categorical: mode imputation → OrdinalEncoder

All preprocessing is embedded inside `sklearn` pipelines to prevent data leakage.

### 5. Models Trained

| Model | Notes |
|---|---|
| **Logistic Regression** | `max_iter=1000`, stratified 75/25 train-test split |
| **Random Forest** | `n_jobs=-1` for parallel training, `random_state=43` |

### 6. Feature Importance

Permutation importance was computed on the Random Forest test set (10 repeats). Top predictors:

1. **`capital-gain`** — mean accuracy drop ~0.048 — investment activity is the clearest signal
2. **`educational-num`** — mean accuracy drop ~0.029 — education level directly drives salary
3. Relationship status, occupation, and hours-per-week followed as secondary predictors

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualization |
| `scikit-learn` | Preprocessing, pipelines, modeling |
| `imbalanced-learn` | Pipeline support for imbalanced datasets |
| `my_utils` (local) | Custom profiling and plotting utilities |

---

## How to Run

1. Make sure `adult-income.csv` is in the same directory as the notebook.
2. Ensure `my_utils.py` is one directory level up (`../my_utils.py`).
3. Install dependencies:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn imbalanced-learn
```

4. Open and run `adult-income.ipynb` top to bottom.

---

## Business Takeaway

The model reveals that income inequality in this dataset is driven by factors that are both measurable and interpretable:

- **Capital gains** reflect access to investment markets — a privilege concentrated in higher income brackets
- **Education** reflects long-term earning potential — each additional year of schooling correlates with higher pay

These are not surprises, but having a model that quantifies and ranks them gives decision-makers a data-backed foundation for policies around education access, tax structure, and income support programs.
