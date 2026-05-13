# 👥 Customer Segmentation — Credit Card Risk Analysis

## What does "Defaulted" mean?
When a customer **defaults**, it simply means they **stopped paying back what they owe** on their credit card — missed payments piled up until the bank wrote off the debt as a loss. Think of it like borrowing money from a friend and never paying them back. From the bank's perspective, that customer is a financial risk.

---

## What is this project about?
This project groups bank customers into **distinct segments** based on their financial profile — age, income, debt, and employment — then examines which group is more likely to default. The goal is to help the bank **offer the right credit card to the right customer**, rather than treating everyone the same.

To do this, we used a machine learning technique called **KMeans Clustering** — an algorithm that automatically finds natural groupings in the data, the same way you might unconsciously sort a pile of coins by size without being told to.

---

## How did we choose the number of groups?

Before clustering, we needed to decide: **how many groups should there be?** We used two standard tests for this.

![Silhouette Score & Inertia](<img width="1189" height="490" alt="silhouette_inertia_score" src="https://github.com/user-attachments/assets/ca78325b-29e1-48a8-94d8-075eccae2a53" />
)

- The **Silhouette Score** (left) peaks at **K=2**, meaning two groups fit the data best.
- The **Elbow Plot** (right) shows no sharp bend, but combined with the silhouette result, **K=2** is the clear choice.
- Conveniently, K=2 also maps naturally to the two outcomes we care about: customers who default and customers who don't.

---

## The Two Customer Groups

![Cluster Plot — Age vs Years Employed]()

The chart above shows customers plotted by **Age** vs. **Years Employed** — two of the most separating features. The two clusters are visually distinct with little overlap.

| | Cluster 0 — The Established Borrowers | Cluster 1 — The Young Starters |
|---|---|---|
| **Age** | 35–60 years old | 20–35 years old |
| **Employment** | Up to 30 years experience | Mostly under 5 years |
| **Income** | High, up to ~$500K | Low, mostly under $100K |
| **Debt** | High card & other debt | Near zero debt |
| **Default Rate** | ~30% | ~25% |

> In plain terms: **Cluster 0** is a financially active, middle-aged group with high incomes *and* high debt. **Cluster 1** is a younger, lower-income group that barely uses credit at all.

---

## Who Actually Defaults More?

![Default Status by Cluster](<img width="571" height="455" alt="output" src="https://github.com/user-attachments/assets/178ca2a7-940c-4dd4-8b27-ed847b09c149" />)

Despite earning significantly more, **Cluster 0 defaults at a higher rate (~30%)** than Cluster 1 (~25%). The reason is straightforward — their aggressive borrowing stretches their finances, making them riskier than their income alone would suggest.

> 💡 **Key Insight:** High income does not equal low risk. Debt behavior matters just as much as earnings.

---

## Recommendations

### 💳 Cluster 0 — Premium Cards, Stricter Checks
Offer **high-limit, rewards-based credit cards** to this group. They're experienced borrowers who will use and value premium products. However, given their higher default rate, pair this with **tighter credit assessments** and responsible limit-setting.

### 🌱 Cluster 1 — Starter Cards, Long-Term Loyalty
Offer **secured or low-limit starter cards** to this group. They're young and just beginning their financial journey — introducing credit responsibly now builds lasting loyalty as their income grows over time.

---

> **Bottom Line:** These two clusters are fundamentally different customers. A one-size-fits-all credit card strategy would under-serve both. Matching the product to the profile maximizes revenue while keeping default risk in check.
