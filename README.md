# **Credit Risk Modeling**

![](img/banner_title.jpeg)

## **1.0 Context**

### **1.1 What Is Credit Risk?**

Credit risk refers to the potential loss a lender faces when a borrower fails to repay a loan or meet contractual obligations. Essentially, it’s the **risk of non-payment of principal and interest**, leading to **disrupted cash flows** and **additional collection costs**. To mitigate this, lenders may charge a higher interest rate (coupon rate), compensating for the increased risk.

Although it’s impossible to predict exactly who will default, **effective credit risk assessment and management** can significantly reduce potential losses. The interest paid by borrowers represents a reward for assuming this risk.

To minimize losses, institutions must have a solid **Credit Risk Management framework**. However, developing one poses significant challenges that are often expensive and time-consuming.
Moreover, in the wake of the COVID-19 pandemic, organizations must strengthen these frameworks — the U.S. Federal Reserve estimated potential pandemic-related loan losses of up to **$700 billion** for major banks under a severe “double-dip” recession scenario.

> **Note:** References are listed at the end of this document.
> Additional details are available in **Section 1** of [this notebook](https://github.com/brunokatekawa/credit_risk/blob/master/Credit_Risk.ipynb).

---

## **2.0 The Challenges**

* **A. Inefficient Data Management:** Difficulty accessing the right data when needed causes significant delays.
* **B. Lack of a Unified Risk Modeling Framework:** Without a centralized structure, institutions struggle to generate consistent risk measures or assess overall exposure.
* **C. Constant Rework:** Inflexible model parameters lead to repetitive efforts, reducing overall efficiency.
* **D. Limited Risk Tools:** Weak analytical tools make it difficult to identify portfolio concentrations or regrade portfolios effectively.
* **E. Manual Reporting:** Spreadsheet-based reporting is slow, error-prone, and resource-intensive.

---

## **3.0 The Solution**

This project addresses **Challenges B, D, and E** by developing a **Credit Risk Model** that processes customer portfolio data (from a `.csv` file), builds a **strategy table**, and estimates the **total expected loss** under the assumption that exposure equals the full loan value and **loss given default (LGD)** equals 100%.
In this setup, a default implies a total loss of the loan amount.

![](img/demo_video.gif)

Live demo: [Credit Risk App on Heroku](https://credit-risk-bk.herokuapp.com/)
*(Note: Loading may take time since the app uses Heroku’s free tier, which hibernates after 30 minutes of inactivity.)*

---

### **3.1 Development Process**

#### **3.1.1 Exploratory Data Analysis**

##### **Descriptive Analysis**

![](img/descriptive-statistics.png)

**Key Insights:**

* Age ranges broadly from 20 to over 100 (outliers later removed).
* Loan amounts vary widely — from **$500** to **$35,000**.
* Some individuals allocate as much as **83% of their income** to loan repayment.
* Employment length and credit history show a mix of early and experienced borrowers.

##### **Hypothesis Map**

This map guided the variable selection and hypothesis testing process.

![](img/hypotheses-map.jpg)

##### **Univariate Analysis**

![](img/univariate-analysis.png)

* **Defaults:** 6,463 (21.94%)
* **Non-defaults:** 22,996 (78.06%)

---

### **3.1.2 Hypothesis Validation — Bivariate Analysis**

#### **Main Hypotheses**

**H3:** Borrowers who default allocate a higher median percentage of income to their loans.
✅ **TRUE** — Defaulting borrowers have higher income-to-loan ratios.
![](img/H3_percentage_income_loan.png)

**H4:** Borrowers with mortgages default more often than renters or owners.
❌ **FALSE** — Renters show the highest default rate.
![](img/H4_home_ownership.png)

**H6:** Borrowers with a previous history of default are more likely to default again.
❌ **FALSE** — Surprisingly, defaults are more common among those without prior default history.
![](img/H6_history_default.png)

**H8:** Personal loans have the highest default rate.
**H9:** Venture loans have the lowest default rate.
![](img/H89_loan_intent.png)
❌ **H8 FALSE**, ✅ **H9 TRUE** — Medical loans show the most defaults; venture loans the least.

**H10:** Higher loan grades correspond to fewer defaults.
❌ **FALSE** — Higher-grade loans do not necessarily have lower default rates.
![](img/H10_loan_grade.png)

---

#### **Hypothesis Summary**

|  ID | Hypothesis                                        | Conclusion |
| :-: | :------------------------------------------------ | :--------: |
|  H1 | More defaults occur among younger borrowers       |   ✅ True   |
|  H2 | Defaulters have lower income than non-defaulters  |   ✅ True   |
|  H3 | Higher income-to-loan ratio for defaulters        |   ✅ True   |
|  H4 | Mortgage holders default the most                 |   ❌ False  |
|  H5 | Longer employment history reduces defaults        |   ✅ True   |
|  H6 | Previous defaults increase risk                   |   ❌ False  |
|  H7 | Longer credit history reduces defaults            |   ✅ True   |
|  H8 | Personal loans have most defaults                 |   ❌ False  |
|  H9 | Venture loans have fewest defaults                |   ✅ True   |
| H10 | Higher loan grades reduce defaults                |   ❌ False  |
| H11 | Defaulted loans have higher median amounts        |   ✅ True   |
| H12 | Defaulted loans have higher median interest rates |   ✅ True   |

---

### **3.1.3 Machine Learning**

#### **Performance Comparison**

![](img/ml_alg_comparison.png)
Highlighted cells indicate the best-performing values per metric.

#### **Confusion Matrices**

![](img/ml_alg_cm.png)
Although **Logistic Regression** shows fewer false negatives, it also has a significantly higher number of false positives (1,217). Considering overall balance, **CatBoostClassifier** performs best.

#### **Final Model Selection**

For this use case, recall is the priority (catching defaulters is more critical than rejecting safe borrowers).
The **CatBoostClassifier** achieves the best compromise between **recall** and **f1-score** (the harmonic mean of precision and recall).

---

### **3.1.4 Business Impact**

The **Confusion Matrix** provides business context:

![](img/confusion_matrix.png)

The key decision is whether to prioritize:

* **(A)** Reducing False Positives — rejecting fewer safe borrowers.
* **(B)** Reducing False Negatives — avoiding missed defaulters.

Here, **Option (B)** is more valuable:

> Missing a default (False Negative) is more costly than rejecting a low-risk borrower (False Positive).

Using this rationale, total expected loss is estimated using:

* **PD (Probability of Default)**
* **LGD (Loss Given Default)**
* **EAD (Exposure at Default)**

![](img/total_expected_loss.png)

**Total Expected Loss:** $19,596,497.74

Without the model, expected losses would exceed $35 million.
Thus, this model **saves approximately $17.8 million** — a significant financial improvement.

---

### **3.1.5 Model Performance — CatBoost Classifier**

#### **Test Set Metrics**

| Precision | Recall | F1-score | ROC AUC | Cohen’s Kappa | Accuracy |
| --------- | ------ | -------- | ------- | ------------- | -------- |
| 0.9535    | 0.7234 | 0.8227   | 0.9367  | 0.7812        | 0.9316   |

#### **Cross-Validation Results (10-Fold Stratified)**

| Model                            | Avg Precision       | Avg Recall      | F1-score            | Avg ROC AUC         |
| -------------------------------- | ------------------- | --------------- | ------------------- | ------------------- |
| CatBoost Classifier              | 0.8796 ± 0.1956     | 0.7190 ± 0.1793 | 0.7837 ± 0.0976     | 0.9041 ± 0.1005     |
| CatBoost (Tuned HP)              | 0.9101 ± 0.1554     | 0.7155 ± 0.1743 | 0.7947 ± 0.0823     | 0.9100 ± 0.0847     |
| CatBoost (Tuned HP + Calibrated) | **0.9625 ± 0.0693** | 0.7060 ± 0.1808 | **0.8100 ± 0.0959** | **0.9094 ± 0.0873** |

Although the tuned model has slightly higher recall, the **Tuned + Calibrated** version provides **better precision, calibration, and f1-score**, making it the optimal choice for deployment in the web app.

---

## **4.0 Next Steps**

* **4.1** Address the remaining challenges:

  * **A:** Improve data accessibility and management.
  * **C:** Automate model parameter tuning to reduce rework.

---

## **References**

* [Lexington Law – Credit History](https://www.lexingtonlaw.com/credit/length-of-credit-history)
* [Investopedia – Credit Risk](https://www.investopedia.com/terms/c/creditrisk.asp)
* [The Crisis of Credit Visualized (YouTube)](https://www.youtube.com/watch?v=bx_LWm6_6tA)
* [Corporate Finance Institute – Credit Risk](https://corporatefinanceinstitute.com/resources/knowledge/finance/credit-risk/)
* [SAS – Credit Risk Management](https://www.sas.com/en_us/insights/risk-management/credit-risk-management.html)
* [McKinsey – Managing Credit Risk Post-COVID](https://www.mckinsey.com/business-functions/risk/our-insights/managing-and-monitoring-credit-risk-after-the-covid-19-pandemic)

