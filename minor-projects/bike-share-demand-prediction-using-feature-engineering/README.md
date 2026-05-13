# 🚲 Bike Share Demand Prediction

## What is this project about?

This project aims to predict **how many bikes will be rented at any given hour** using historical data from a bike-sharing service. Think of it like forecasting demand — the same way a coffee shop predicts how many cups it will sell on a rainy Monday morning.

To do this, we trained two types of machine learning models and compared their performance — one with raw data, and one where we put extra effort into preparing and enriching the data before training.

---

## What is Feature Engineering?

**Feature engineering** is the process of transforming raw data into more meaningful inputs for a model.

For example, instead of feeding the model a raw timestamp like `2011-07-11 09:00:00`, we broke it down into:
- 📅 **Month** → July
- 📆 **Day** → Monday
- 🕘 **Hour** → 9

This gives the model much more useful context — it can now learn that *Monday mornings in July* tend to be busier than *Sunday nights in January*.

We also:
- Converted temperatures from **Celsius to Fahrenheit**
- Created a **temperature variance** column showing how much warmer or colder it was compared to the seasonal average
- Removed columns that were redundant or would cause data leakage

---

## The Two Models

We trained a **Random Forest** model — a powerful algorithm that makes predictions by combining the results of hundreds of decision trees — under two different conditions:

| | Baseline Model | Enhanced Model |
|---|---|---|
| Data Used | Raw, unprocessed | Cleaned & feature-engineered |
| Datetime Handling | Left as-is | Broken into month, day & hour |
| Temperature | Celsius | Fahrenheit + variance added |

---

## Results

### 🔴 Baseline Model (No Feature Engineering)

| Metric | Training | Test |
|--------|----------|------|
| R²     | 0.846    | 0.296 |
| RMSE   | 71.188   | 151.993 |
| MAE    | 47.729   | 107.199 |

> The model performed well on data it had already seen, but struggled badly on new, unseen data. This is called **overfitting** — the model memorized the training data instead of learning general patterns.

---

### 🟢 Enhanced Model (With Feature Engineering)

| Metric | Training | Test |
|--------|----------|------|
| R²     | 0.976    | 0.843 |
| RMSE   | 27.938   | 71.825 |
| MAE    | 18.282   | 48.399 |

> A dramatic improvement. The model now generalizes well to new data, explaining **84% of the variation** in bike rentals.

---

## Baseline vs. Enhanced — Side by Side

| Metric | Baseline (Test) | Enhanced (Test) | Improvement |
|--------|----------------|-----------------|-------------|
| R²     | 0.296          | 0.843           | +0.547 ✅  |
| RMSE   | 151.993        | 71.825          | -80.168 ✅ |
| MAE    | 107.199        | 48.399          | -58.800 ✅ |

> 💡 **What do these numbers mean?**
> - **R²** — How well the model explains the data. Closer to 1.0 is better.
> - **RMSE** — The average prediction error in number of bikes. Lower is better.
> - **MAE** — Similar to RMSE but less sensitive to large errors. Lower is better.

---

## Key Takeaway

> Feature engineering was the **single most impactful factor** in this project. Simply breaking down the timestamp into hour, day, and month allowed the model to discover that time of day and day of the week are the strongest drivers of bike rental demand — something the raw data completely obscured.
