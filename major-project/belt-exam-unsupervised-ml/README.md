<div align="center">

# 🎓 Student Success

### Clustering & Predictive Modeling of Online Learners

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

*An End-to-End Machine Learning Report — Version B (Intermediate ML)*

</div>

---

## 📑 Table of Contents

- [🎯 The Problem](#-the-problem)
- [🗂️ The Data](#️-the-data)
- [🧩 Part 1 — Student Segmentation (Unsupervised)](#-part-1--student-segmentation-unsupervised)
  - [1️⃣ Exploratory Data Analysis & Key Outcomes](#1️⃣-exploratory-data-analysis--key-outcomes)
  - [2️⃣ Silhouette Score & Elbow — Choosing k](#2️⃣-silhouette-score--elbow--choosing-k)
  - [3️⃣ Heatmap — Average Feature Values per Cluster](#3️⃣-heatmap--average-feature-values-per-cluster)
  - [4️⃣ Categorical Features by Cluster](#4️⃣-categorical-features-by-cluster)
  - [5️⃣ Numerical Features by Cluster](#5️⃣-numerical-features-by-cluster)
- [🤖 Part 2 — Predicting Pass / Fail (Supervised)](#-part-2--predicting-pass--fail-supervised)
- [🏁 Conclusion](#-conclusion)

---

## 🎯 The Problem

> [!IMPORTANT]
> **An online school wants to do two things, and getting them right has real consequences for students.**

The school provided data on its learners and asked for **two deliverables**:

| # | Task | Type | Why it matters |
|:-:|------|------|----------------|
| 1️⃣ | **Segment students into groups** and describe them | Unsupervised | Understand *what kinds* of students exist |
| 2️⃣ | **Predict who will fail** the course | Supervised | Intervene **before** it's too late |

The whole point is **early intervention**. If we can flag a struggling student *before* they fail, the school can step in — tutoring, counseling, support — and change the outcome. A model that catches at-risk students is therefore not just an academic exercise; **it directly protects students from falling through the cracks.**

> [!NOTE]
> **IMD** = *Index of Multiple Deprivation* — a combined measure of socioeconomic deprivation for a student's neighborhood. **Higher = less deprived**, **lower = more deprived**.

---

## 🗂️ The Data

Two related datasets, same **12,737 students**:

| Dataset | Purpose | Shape | Notes |
|---------|---------|:-----:|-------|
| `Option_B_clustering.csv` | Segmentation | **12,737 × 13** | Compact, fully numeric student profile |
| `Option_B_modeling.csv` | Prediction | **12,737 × 931** | Wide — daily activity, assessment timing, course enrollment |

The school's extra tracking (daily logins, how early/late assessments were submitted, which courses were taken) is what blew the modeling table up to **931 columns**. Target variable: `passed_course`.

---

## 🧩 Part 1 — Student Segmentation (Unsupervised)

### 1️⃣ Exploratory Data Analysis & Key Outcomes

Before any modeling, every feature was inspected. The headline news is that **the data was remarkably clean**:

- ✅ **No missing values** across all 13 columns
- ✅ Only **7 duplicate rows** — removed, leaving **12,730** unique students
- ✅ All features numeric and ready for distance-based clustering

**What the simple analysis actually told us:**

<div align="center">
  <img width="790" height="390" alt="pass_fail" src="https://github.com/user-attachments/assets/5116df2d-ae82-4891-8a78-9775633a53ea" />
</div>

> **🔑 Outcome — class imbalance.** Roughly **76% of students passed** and **24% failed**. This single chart is the most consequential EDA finding: it warns us up front that the prediction model (Part 2) will face an **imbalanced target**, which later justifies watching *recall* closely and testing SMOTE.

<div align="center">
  <img width="790" height="390" alt="output" src="https://github.com/user-attachments/assets/63755dbc-9313-459b-90f7-b9ef5ec5d1f4" />

</div>

> **🔑 Outcome — study load is skewed.** Most students cluster around a **light credit load (~60)**, with a long tail of heavily-loaded students. That spread turns out to be the *defining* axis separating the two clusters later on.

<div align="center">
  <img width="790" height="389" alt="output" src="https://github.com/user-attachments/assets/6fd93ff6-6871-4f7f-b1ed-aec863f89929" />

</div>

> **🔑 Outcome — assessment scores carry the signal.** Scores spread widely across the population — the richest, most discriminating feature group, and exactly what KMeans latches onto.

---

### 2️⃣ Silhouette Score & Elbow — Choosing k

Numeric features were **standardized** (so no single feature dominates the distance math), then KMeans was fitted for **k = 2 … 10**. Two diagnostics were used **together**:

<div align="center">
  <img width="1189" height="490" alt="output" src="https://github.com/user-attachments/assets/3e5674f1-fa0a-4a4d-9739-672bd18cc7ea" />
</div>

> [!TIP]
> **The silhouette score is the decisive evidence here.** 🎯 It peaks sharply at **k = 2** (≈ **0.17**) and drops immediately for every larger *k*. The elbow plot agrees — the bend sits at **k = 2**. When inertia and silhouette point to the *same* number, the choice is well-justified.

**✅ Decision: k = 2.** It's also intuitive — students split naturally into **those who tend to pass** and **those who tend to struggle**.

---

### 3️⃣ Heatmap — Average Feature Values per Cluster

This is where the two groups reveal their personalities. 🔥

<div align="center">
  <img width="1056" height="390" alt="output" src="https://github.com/user-attachments/assets/c355844f-9fa1-46c4-910c-1d535530444b" />
</div>

**Cluster 0 scored higher on *all five* assessments** than Cluster 1. The most revealing contrast is **study load vs. performance**:

| Cluster | 📚 Studied Credits | 📝 Assessment Scores |
|:-------:|:------------------:|:--------------------:|
| **🟢 Cluster 0** | **Lower** (~71.6) | **Higher** (≈ 80–86) |
| **🔴 Cluster 1** | **Higher** (~80.9) | **Lower** (≈ 61–66) |

> [!IMPORTANT]
> **💡 The core insight — Quality over Quantity.** Cluster 0 carried a **lighter load** and could focus on each course → stronger scores. Cluster 1 took on **more credits**, the heavier workload split their attention → weaker scores. **Fewer courses, done well, beats more courses spread thin.**

---

### 4️⃣ Categorical Features by Cluster

<div align="center">
 <img width="1789" height="990" alt="image" src="https://github.com/user-attachments/assets/15511a56-b9d8-4907-ae1c-a7b4064297a0" />
</div>

> [!NOTE]
> No data dictionary was provided, so categories appear as **numeric codes** — we read *trends*, not exact labels.

**What stands out:**
- 🎯 **`passed_course` is the cleanest split** — Cluster 0 (purple) heavily dominates the *passed* group, while Cluster 1 (yellow) makes up a large share of the *failed* group. The clustering essentially **rediscovered the pass/fail divide on its own**, with no labels given.
- Demographics (**gender, age band, IMD band, disability**) are **fairly evenly mixed** across both clusters — meaning the split is driven by **behavior and performance**, *not* by who the student is. That's an encouraging, fair result.

---

### 5️⃣ Numerical Features by Cluster

<div align="center">
  <img width="1989" height="990" alt="image" src="https://github.com/user-attachments/assets/9c03e54d-8dbd-425d-8486-2d7851c8c318" />
</div>

The violins show **both the level and the consistency** of each group:

| | 🟦 Cluster 0 — *Focused & Consistent* | 🟩 Cluster 1 — *Overloaded & Inconsistent* |
|---|---|---|
| **Shape** | Tight, narrow around a high mean | Wide, stretched, lower mean |
| **Scores** | Consistently **high** across all 5 assessments | **Lower** and far more **variable** |
| **Story** | Stable, reliable performers | Uneven results, signs of strain |

> **💡 Read-out:** Across **every** assessment, Cluster 0 sits higher *and* tighter. Cluster 1's wide spread is the statistical fingerprint of a heavier course load dividing attention — the same quality-vs-quantity story, now visible in the *variance*, not just the average.

---

## 🤖 Part 2 — Predicting Pass / Fail (Supervised)

Part 2 follows the same logic, but the punchline is simpler: **several models were built, and one was chosen — with recall as the deciding metric.**

### 🛠️ Preparing the Wide Data

- **12,737 × 931**, no duplicates, no nulls
- Target imbalanced (**~76% pass / 24% fail**) — exactly the warning EDA gave us
- **Standardized** → then **PCA** (`n_components=0.9`) compressed **900+ features → 341 components**, keeping **90% of the variance**

### 🔁 Three Models, Each Fixing the Last

| Model | Technique | Result |
|:-----:|-----------|--------|
| **1️⃣ Baseline** | No regularization | 🔴 Severe overfitting — 100% train vs ~84% val |
| **2️⃣ + Dropout** | Dropout 0.3 / 0.2 | 🟡 Less overfit (~95% train), val still ~86% |
| **3️⃣ + EarlyStopping** | Dropout **+** EarlyStopping | 🟢 **Best balance — chosen as final** |

<div align="center">
  <img width="1189" height="989" alt="image" src="https://github.com/user-attachments/assets/85daa903-ef3b-4fdf-a754-ebbfa316fdb0" />
  <br><em>Model 1 — textbook overfitting</em>
  <br><br>
  <img width="1189" height="989" alt="image" src="https://github.com/user-attachments/assets/fb371aa5-c1b7-4248-ac09-e1bc096bf3c8" />
  <br><em>Model 3 — EarlyStopping closes the gap (final model)</em>
</div>

**Model 3** stopped automatically after **~8 epochs**, kept the train/val gap small, and stabilized validation loss around **0.38–0.41**. A **SMOTE** experiment (rebalancing to 50/50) was also tested but traded away recall, so the primary Model 3 was kept.

### 🎯 Final Results — and Why Recall Rules

| Metric | Score | Meaning |
|--------|:-----:|---------|
| **Accuracy** | ≈ **86%** | Right ~86 times out of 100 |
| **Precision** | ≈ **88%** | A flagged student is usually truly at risk |
| **🌟 Recall** | ≈ **93–95%** | **Catches almost all students who actually fail** |

> [!IMPORTANT]
> **Why we obsess over recall, not accuracy.** 🎯
> A **missed** at-risk student = a struggling kid who gets **no help**. A **false alarm** = a passing student who's simply offered extra support — harmless.
> The cost of the two mistakes is **wildly unequal**, so we optimize for the one that hurts: **recall**. Catching ~93% of failing students is precisely the behavior this problem demands.

---

## 🏁 Conclusion

This project ran the full pipeline — from raw data to two working deliverables:

- 🧩 **Clustering** uncovered a clear **quality-vs-quantity** story: lighter-load students (Cluster 0) performed consistently well, while overloaded students (Cluster 1) struggled with uneven results. The split was driven by **behavior, not demographics** — and the model rediscovered the pass/fail line unsupervised.
- 🤖 **Modeling** delivered a Dropout + EarlyStopping network on PCA-reduced features, reaching **~86% accuracy with ~93% recall** — tuned deliberately to **catch at-risk students**.

> [!TIP]
> No model is perfect, but **high accuracy + high recall** makes this a **reliable early-warning tool** — letting the school act *proactively* instead of reactively, and supporting the students who need it most. 🎓

<div align="center">

⭐ *If this report helped, give the repo a star!* ⭐

</div>
