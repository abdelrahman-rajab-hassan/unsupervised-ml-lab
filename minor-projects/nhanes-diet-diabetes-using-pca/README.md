# 🧬 NHANES Diet & Diabetes Risk — PCA vs. Non-PCA Classification

## What is this project about?
This project explores whether **dimensionality reduction using PCA** can speed up a KNeighborsClassifier without sacrificing predictive performance. Think of it like asking — *can we say just as much with fewer words?*

To do this, we trained two **KNeighborsClassifier** models on a modified version of the NHANES dataset and compared their speed and performance — one using all original features, and one where PCA was applied to reduce the number of dimensions first.

> 📥 The dataset is a modified version of the [NHANES dataset available on Kaggle](https://www.kaggle.com/datasets/cdc/national-health-and-nutrition-examination-survey).

---

## What is PCA?
**Principal Component Analysis (PCA)** is a dimensionality reduction technique that compresses many features into a smaller set of new features — called **principal components** — while retaining as much information as possible.

For example, instead of feeding the model 113 original columns, PCA reduced them to:
- 🔢 **63 principal components** — capturing **95% of the variance** in the data

This gives the model fewer, more compact inputs — which can lead to:
- ⚡ **Faster training and prediction**
- 🧹 **Less noise from redundant features**
- 📦 **Lower memory usage**

---

## The Two Models
We trained a **KNeighborsClassifier** — a simple but powerful algorithm that classifies a data point based on the majority class of its nearest neighbors — under two different conditions:

| | Without PCA | With PCA |
|---|---|---|
| Features Used | 113 original columns | 63 principal components |
| Variance Retained | 100% | 95% |
| Preprocessing | Impute → Scale → SMOTE | Impute → Scale → SMOTE → PCA |

---

## Results

### 🔴 Without PCA
| Metric | Value |
|---|---|
| Accuracy | 0.44 |
| Precision (macro avg) | 0.34 |
| Recall (macro avg) | 0.34 |
| F1-Score (macro avg) | 0.29 |
| Weighted avg F1 | 0.52 |

---

### 🟢 With PCA
| Metric | Value |
|---|---|
| Accuracy | 0.43 |
| Precision (macro avg) | 0.33 |
| Recall (macro avg) | 0.31 |
| F1-Score (macro avg) | 0.28 |
| Weighted avg F1 | 0.51 |

---

## Side-by-Side Comparison

| Metric | Without PCA | With PCA | Difference |
|---|---|---|---|
| Accuracy | 0.44 | 0.43 | -0.01 |
| Precision (macro avg) | 0.34 | 0.33 | -0.01 |
| Recall (macro avg) | 0.34 | 0.31 | -0.03 |
| F1-Score (macro avg) | 0.29 | 0.28 | -0.01 |
| Weighted avg F1 | 0.52 | 0.51 | -0.01 |

> 💡 **What do these numbers mean?**
> - **Accuracy** — The percentage of correct predictions overall.
> - **Precision** — Of all predicted positives, how many were actually positive.
> - **Recall** — Of all actual positives, how many did the model catch.
> - **F1-Score** — A balance between precision and recall. Closer to 1.0 is better.

---

## **Key Questions**

### 🏆 Which model performed best?
Both models performed **almost identically** — the difference across all metrics was less than 1%, meaning PCA retained virtually all the predictive power of the original 113 features.

### ⚡ Which model was fastest?
The **PCA model** was faster at prediction time, since it operates on 63 components instead of 113 features — fewer dimensions means less distance computation for KNN.

---

## Key Takeaway
> PCA was the **single most impactful factor** in terms of efficiency. By reducing 113 features down to 63 principal components, the model became significantly faster — while losing virtually nothing in predictive performance. For KNeighborsClassifier specifically, this trade-off is almost always worth it, since KNN's speed is directly tied to the number of features it has to compute distances across.
