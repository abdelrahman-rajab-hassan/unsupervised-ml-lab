# 🛍️ Customer Segmentation with K-Means Clustering

> **Who are my customers — and how should I talk to each of them?**
> This project answers that question using purchase history data and a machine learning technique called K-Means clustering.

---

## 🧩 The Problem

As an online retailer, sending the same message to every customer is a missed opportunity. A loyal VIP buyer and someone who shopped once two years ago deserve completely different conversations.

Using **RFM data** — three simple but powerful signals about customer behavior — we grouped customers into meaningful segments and designed targeted marketing actions for each one.

---

## 📐 What is RFM?

| Letter | Stands for | What it means |
|--------|-----------|----------------|
| **R** | Recency | How recently did the customer buy? |
| **F** | Frequency | How often do they buy? |
| **M** | Monetary Value | How much do they spend? |

The better a customer scores on all three, the more valuable they are to the business.

---

## ⚙️ How It Works (Simply Put)

1. **Prepare the data** — Load customer purchase history and clean it up.
2. **Scale the numbers** — Make sure Recency, Frequency, and Monetary Value are on the same scale so no single one dominates the math.
3. **Find the right number of groups** — Use two methods to decide how many clusters make sense:
   - 📉 **Elbow Method** — looks for the point where adding more groups stops helping much.
   - 📊 **Silhouette Score** — measures how well-separated the groups are.
4. **Run K-Means** — The algorithm automatically groups similar customers together.
5. **Understand each group** — Analyze the characteristics of each cluster and give them meaningful names.
6. **Build a marketing plan** — Suggest specific actions for each customer segment.

---

## 🔍 Finding the Right Number of Groups

<img width="1189" height="490" alt="output" src="https://github.com/user-attachments/assets/10ece77b-a66e-43b7-a23c-c6e2448b68a1" />


Both methods pointed to **K = 4** as the sweet spot:
- The Elbow chart shows the curve bending noticeably at 4.
- The Silhouette score peaks at 4–5, with 4 being the cleaner choice.

---

## 👥 The 4 Customer Segments

<img width="490" height="490" alt="outpdut" src="https://github.com/user-attachments/assets/370d805e-1527-4e53-86a7-c4016ab36698" />


| Segment | % of Customers | Last Purchase | Visit Freq. | Avg. Spend |
|---------|---------------|--------------|------------|------------|
| 🛒 Occasional Buyers | **72.8%** | ~42 days ago | 4 visits | $1,567 |
| 💤 Lapsed / Churned | **24.8%** | ~247 days ago | 2 visits | $557 |
| ⭐ Frequent Buyers | **2.2%** | ~13 days ago | 35 visits | $21,820 |
| 👑 Champions / VIPs | **0.2%** | ~7 days ago | 65 visits | $184,144 |

---

## 🎯 Marketing Actions by Segment

### 🛒 Occasional Buyers — *72.8% of customers*
Shopped recently but not very often. They're interested — just not hooked yet.

**Goal:** Turn them into regulars.
- Send personalized product recommendations based on past purchases
- Offer a "come back" discount after 30 days of inactivity
- Run loyalty program onboarding campaigns
- Use email/SMS nudges like *"You might also love…"*

---

### 💤 Lapsed / Churned — *24.8% of customers*
Haven't bought in months. Low frequency, low spending. They may have moved on.

**Goal:** Win them back — or let them go gracefully.
- Launch a win-back email campaign: *"We miss you! Here's 20% off"*
- Highlight new arrivals or what's changed since they left
- If they don't re-engage after 2–3 attempts, suppress from active lists to save budget
- Survey them: *"What made you stop shopping with us?"*

---

### ⭐ Frequent Buyers — *2.2% of customers*
Shop often and spend meaningfully. A small but high-value group on the rise.

**Goal:** Nurture them toward VIP status.
- Enroll them in a tiered loyalty rewards program
- Give early access to sales and new products
- Send personalized thank-you messages
- Offer bundle deals or subscription options that reward their frequency

---

### 👑 Champions / VIPs — *0.2% of customers*
The crown jewels. Shop constantly, spend the most, and are deeply engaged.

**Goal:** Protect and delight them at all costs.
- Assign a dedicated account manager or concierge-style service
- Invite to exclusive events, private sales, or beta product launches
- Send handwritten thank-you notes or surprise gifts
- Never let them feel like just another customer — they aren't

---

## 📊 Cluster Breakdown

<img width="1487" height="362" alt="outut" src="https://github.com/user-attachments/assets/54514f9a-7b2e-4cf5-9506-b7bff61b4bbe" />

The boxplots above show how Recency, Frequency, and Monetary Value are distributed across the four clusters — making the differences between segments clearly visible.

---

## 🛠️ Tools Used

- **Python** — the programming language
- **Pandas & NumPy** — for data handling
- **Scikit-learn** — for K-Means clustering and scaling
- **Matplotlib & Seaborn** — for charts and visualizations

---

## 📁 File Structure

```
📦 project
 ┣ 📓 KMeans_Exercise.ipynb   ← the full analysis notebook
 ┣ 📄 rfm.csv                 ← customer purchase data
 ┗ 📄 README.md               ← you are here
```

---

*Built as part of a customer analytics exercise. The goal: turn raw transaction data into actionable business strategy.* 🚀
