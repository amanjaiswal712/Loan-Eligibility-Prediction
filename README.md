# Credit Risk Analysis and Prediction

## Project Overview
This repository contains a comprehensive evaluation and predictive modeling pipeline for financial credit risk. The core objective is to execute a binary classification task: distinguishing accurately between "good" credit risks (highly likely to honor repayment) and "bad" credit risks (high probability of default). 

By transitioning from manual underwriting to algorithmic statistical analysis and machine learning, this project aims to optimize the detection of actual risk, minimize costly Type I errors (approving a bad risk), and maintain a commercially viable volume of loan approvals while ensuring strict regulatory transparency.

---

## Dataset Architecture
The empirical foundation of this analysis is the **German Credit Dataset** (originally prepared by Prof. Hofmann, University of Hamburg). The dataset has been mathematically refined into a highly interpretable structure containing **1000 distinct instances** and **10 primary columns** (9 independent predictors, 1 binary target variable).

### Target Variable (`Risk`)
* **Class Imbalance:** 70% (700 applicants) are classified as "good" risk (0), while 30% (300 applicants) are classified as "bad" risk (1). 

### Key Predictor Variables
* **Continuous:** `Age`, `Credit amount` (DM), `Duration` (months).
* **Categorical:** `Sex`, `Job` (Ordinal 0-3), `Housing`, `Saving accounts`, `Checking account`, `Purpose`.
* **Missing Data:** 183 missing entries in `Saving accounts` and 394 in `Checking account`. These were strategically imputed as "unknown" to capture the predictive weight of applicants operating outside standard retail banking.

---

## Exploratory Data Analysis (EDA) & Key Statistics

Extensive exploratory data analysis revealed critical non-parametric relationships and risk concentrations within the portfolio:

* **Central Tendencies:** The average applicant is 35.5 years old, seeking a mean credit amount of 3,271.25 units over an average duration of 20.90 months.
* **Distribution Skewness:** `Credit amount` exhibits severe right-skewness (median: 2319.50 vs. max: 18,424.00), necessitating logarithmic transformation to stabilize variance.
* **Transactional Liquidity (The Ultimate Discriminator):** * Applicants with **"little"** checking account balances possess a catastrophic **49.27% bad risk rate**.
  * Applicants with **"unknown"** checking accounts (imputed from NaNs) possess an extraordinarily low default rate of just **11.67%**, highlighting a highly capitalized borrower profile bypassing standard retail checking.
* **Housing Status:** Homeowners exhibit a 26.08% bad risk rate, compared to renters at 39.10% and "free" housing at 40.74%.
* **Loan Purpose:** Unsecured experiential loans ("vacation/others") carry the highest bad risk proportion (**41.67%**), while asset-backed or smaller consumption loans like "radio/TV" exhibit the lowest (**22.14%**).

---

## Inferential Statistical Diagnostics

Rigorous hypothesis testing was systematically applied to validate observed correlations:

* **Chi-Square Tests (Categorical):** * `Checking account` generated a massive Chi-Square statistic of 123.72 (p-value: **1.21e-26**), proving it is the most statistically potent categorical discriminator.
  * `Job` classification failed to achieve statistical significance (p = 0.5965).
* **Independent Samples T-Tests (Continuous):**
  * `Duration`: Good risks average 19.2
