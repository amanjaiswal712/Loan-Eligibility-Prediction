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
  * `Duration`: Good risks average 19.28 months vs. bad risks averaging 24.86 months (p-value: **2.40e-10**).
  * `Credit amount`: Good risks average 2985.45 DM vs. bad risks averaging 3938.12 DM (p-value: **2.47e-05**).
* **ANOVA:** The relationship between `Purpose` and `Credit amount` strongly rejected the null hypothesis (p-value: **1.58e-16**), confirming that capital request volumes are structurally dependent on the loan's intended use.

---

## Feature Engineering Pipeline

To maximize gradient descent efficiency and algorithmic accuracy, the following advanced features were synthesized:

1. **`Age_group`**: Continuous age partitioned into categorical bins (18-30, 31-45, 46-60, 60+) to capture non-linear generational thresholds.
2. **`Credit_per_month`**: A synthetic Debt-to-Income (DTI) vector calculated as `Credit amount / Duration`.
3. **`Log_Credit_amount`**: A logarithmic transformation applied to the `Credit amount` to normalize the heavy-tailed distribution and prevent outlier leverage.

*Data was split using 80/20 stratified sampling, processed via a `ColumnTransformer` pipeline utilizing `StandardScaler` for numeric variables and `OneHotEncoder` for categorical variables.*

---

## Algorithmic Architecture: Champion-Challenger

The project leverages the financial industry-standard "Champion-Challenger" paradigm to balance regulatory transparency with predictive power.

### 1. The Champion: Logistic Regression
* **Description:** A highly interpretable, regulatory-compliant statistical model estimating the log-odds of default via Maximum Likelihood Estimation (MLE).
* **Advantage:** Enables extraction of explicit "adverse action reasons" for strict Consumer Financial Protection Bureau (CFPB) compliance.

### 2. The Challenger: Random Forest Classifier
* **Description:** A sophisticated, non-linear ensemble algorithm utilizing Bootstrap Aggregating (Bagging) and Feature Subspace Sampling across 200 decision trees.
* **Advantage:** Captures highly complex, non-linear feature interactions without manual intervention.

---

## Model Evaluation

The models were blindly evaluated against a 200-instance holdout test set. 

| Metric | Logistic Regression (Champion) | Random Forest (Challenger) |
| :--- | :--- | :--- |
| **Global Accuracy** | 75.0% | 76.0% |
| **ROC-AUC** | **0.779** | **0.792** |
| **Bad Risk Recall** | 37.0% | 23.0% |
| **Bad Risk Precision** | 65.0% | 88.0% |

**Insights:** * The **Random Forest** achieved superior rank-ordering capability (AUC: 0.792) and operates highly conservatively—when it triggers a denial, it possesses an 88% certainty of being correct.
* The **Logistic Regression** model provides a better baseline for catching a wider net of bad risks (higher recall) while maintaining the linear transparency required for regulatory environments.

---

## Feature Importance Extraction

Dissecting the models reveals the fundamental mathematical drivers of credit risk:

* **Top Escalating Factors (Higher Risk):**
  * `Checking account_little` (LR Coef: +0.657, RF Gini: 0.073)
  * `Duration` (LR Coef: +0.573, RF Gini: 0.106)
  * `Purpose_education` (LR Coef: +0.648)
* **Top Buffering Factors (Lower Risk):**
  * `Checking account_unknown` (LR Coef: -0.965, RF Gini: 0.146 - *Highest overall importance in both models*)
  * `Log_Credit_amount` (LR Coef: -0.456)
* **Random Forest Specifics:** The RF heavily prioritized the engineered continuous metrics, identifying specific non-linear breakpoints in `Credit_per_month` (Importance: 0.107) and `Duration`.

---

## Strategic Business Implications

1. **Architectural Deployment:** Deploy the Logistic Regression model as the primary production engine to ensure CFPB compliance and explainability. Run the Random Forest in parallel "shadow mode" to continuously benchmark the scorecard and flag highly non-linear risk segments.
2. **Liquidity-Based Policy Revisions:** Enforce dramatically stricter underwriting standards, higher collateral requirements, or lower credit limits for applicants triggering the "little" checking account threshold.
3. **Duration Risk Pricing:** Implement aggressive risk-based pricing models that attach progressively higher interest rate premiums to any loan extending beyond a 24-month horizon to offset compounding default probabilities.
4. **Mandatory DTI Proxies:** Integrate the engineered `Credit_per_month` ratio into automated decision APIs, shifting focus from gross credit amounts to individualized monthly financial burden analysis.
