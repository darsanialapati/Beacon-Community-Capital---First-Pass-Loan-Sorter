# Beacon Community Capital — Loan Approval Decision Support

### Logistic Regression Classification Model

## Project Overview

Beacon Community Capital is a Community Development Financial Institution (CDFI) based in Columbus, Ohio. The organization serves borrowers who may be underserved by traditional financial institutions, including first-time borrowers, immigrants, and local business owners rebuilding after financial setbacks.

As loan application volume increased, Beacon needed a more scalable way to support its loan officers while keeping lending decisions:

- Consistent
- Explainable
- Efficient
- Human-accountable

I built a reproducible machine learning workflow that uses historical loan application data to support loan officers by triaging applications into confidence bands.

The model is designed as a **decision-support tool**, not as a replacement for human lending decisions.

---

## Business Problem

Beacon's loan application volume increased significantly while the number of loan officers remained unchanged.

This created several operational challenges:

- Application volume increased from approximately **500 to 1,800 applications per month**
- Five loan officers continued to manually review every application
- Average decision times increased to approximately **3–4 business days**
- Manual decisions could vary between officers
- Decision reasoning was not always consistently documented
- A simple FICO-based rule did not adequately represent Beacon's broader lending approach

Beacon had previously explored a basic rule:

```text
Approve if FICO score >= 660
Reject if FICO score < 660
```

While useful for identifying some straightforward cases, this approach was too simplistic.

An applicant with a lower credit score could still have other strong characteristics such as:

- Stable income
- Significant savings
- Longer employment history
- Reasonable loan amount
- Favorable overall financial profile

A broader model was therefore needed to consider multiple application characteristics together.

---

## Objective

I built a **logistic regression classifier** to estimate the historical approval decision Beacon's loan officers would most likely have made.

The model predicts:

> **Historical loan approval decisions**

It does **not** predict:

> **Whether a borrower will repay or default on a loan**

For every application, the model generates an approval probability between `0` and `1`.

A higher probability means the application more closely resembles applications that Beacon historically approved.

A lower probability means the application more closely resembles applications that Beacon historically rejected.

---

## Decision-Support Framework

The predicted probability is converted into three business-defined confidence bands.

| Predicted Approval Probability | Decision-Support Action |
|---|---|
| `p >= 0.80` | Expedited workflow — likely approval |
| `0.20 <= p < 0.80` | Mandatory loan officer review |
| `p < 0.20` | Likely rejection — still requires human review |

These thresholds are **business-defined rules**, not thresholds automatically learned by the model.

The model never makes a final rejection decision independently.

---

## Dataset

**File:** `loan_applications.csv`

Each row represents one historical loan application.

### Key Columns

| Category | Columns |
|---|---|
| Identifier | `applicant_id` |
| Credit | `fico_score` |
| Financial | `annual_income`, `loan_amount`, `savings_balance` |
| Loan | `loan_term_months` |
| Employment | `employment_status`, `years_employed` |
| Target | `approved` |

`applicant_id` is used only as an identifier and is never included as a model feature.

---

## Workflow

The analysis follows four main stages:

1. **Load and clean the data**
2. **Explore historical approval patterns**
3. **Train a logistic regression model**
4. **Evaluate and interpret model performance**

---

## Technology Stack

The workflow was developed in Python using:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
```

### Recommended Environment

- Python 3.10+
- VS Code
- Jupyter extension for VS Code

---

## How to Run

1. Clone or download the repository.
2. Open the project in VS Code.
3. Open:

```text
Learner_Guide_Notebook.ipynb
```

4. Select a Python environment containing the required dependencies.
5. Run the notebook from top to bottom.

---

# 1. Data Cleaning

Historical loan records contained several data-quality issues that needed to be addressed before modeling.

The notebook documents each cleaning step and reports record counts before and after the transformations.

## Currency Conversion

Several financial variables were stored as text values such as:

```text
"$53,000"
```

These columns were converted to numeric values:

- `annual_income`
- `loan_amount`
- `savings_balance`

---

## Duplicate Removal

Exact duplicate records were removed to prevent duplicated applications from disproportionately influencing the model.

---

## Invalid Value Handling

Records containing clearly impossible values were removed.

Examples included:

```text
fico_score > 850
annual_income < 0
loan_amount <= 0
loan_term_months > 84
years_employed > 60
savings_balance < 0
```

Missing FICO values were not treated as invalid records.

---

## Missing FICO Scores

Missing values in `fico_score` were filled using the **median FICO score** from the dataset.

Median imputation was used because it is less sensitive to extreme values than the mean.

---

# 2. Exploratory Data Analysis

Exploratory analysis was used to better understand Beacon's historical lending decisions and the relationships between application characteristics and approval outcomes.

The notebook includes the following analyses.

## Approval Distribution

Historical approved and rejected applications are compared using:

- Record counts
- Bar chart visualization

This provides an understanding of the target-class distribution before modeling.

---

## FICO Score Distribution

A histogram is used to examine the distribution of applicant FICO scores.

A vertical reference line is added at:

```text
FICO = 660
```

This helps visualize Beacon's previous single-score approval rule relative to the actual applicant population.

---

## Approval Rate by Employment Status

Approval rates are calculated across different employment categories.

This analysis helps identify whether historical approval patterns differed across employment groups.

---

## FICO Score vs. Approval Decision

A box plot compares FICO scores for historically:

- Approved applications
- Rejected applications

The overlap between the two groups demonstrates why a single credit-score threshold does not fully explain historical decisions.

---

## Correlation Analysis

A correlation heatmap is generated across numeric variables and the approval outcome.

This provides an initial view of relationships between:

- Credit score
- Income
- Loan amount
- Loan term
- Employment history
- Savings
- Historical approval decisions

Correlation is treated as an association and not evidence of causality.

---

# 3. Modeling Approach

## Target Engineering

The original approval field is converted into a binary modeling target:

```text
Approved = 1
Rejected = 0
```

The resulting variable is:

```python
approved_flag
```

---

## Feature Engineering

The model uses the following numeric application characteristics:

```python
fico_score
annual_income
loan_amount
loan_term_months
years_employed
savings_balance
```

A simple employment indicator is also created:

```python
is_self_employed
```

where:

```text
1 = Self-employed
0 = Otherwise
```

---

## Leakage Prevention

The following variables are explicitly excluded from the feature matrix:

```python
applicant_id
approved
approved_flag
```

This prevents identifiers and the target itself from leaking information into the model.

---

## Train/Test Split

The data is divided into training and testing datasets using:

```python
test_size=0.2
stratify=y
random_state=42
```

This results in:

- **80% training data**
- **20% test data**

Stratification helps preserve the approved/rejected class distribution across both datasets.

---

## Logistic Regression Model

The primary model is:

```python
LogisticRegression(max_iter=1000)
```

Logistic regression was selected because it provides:

- Probabilistic predictions
- Relatively straightforward interpretation
- A transparent baseline for binary classification
- Coefficients that help explain relationships between model inputs and historical approval decisions

---

# 4. Model Interpretation

## Raw Coefficients

The notebook plots the logistic regression coefficients to examine how each feature is associated with the model's predicted historical approval probability.

Because the features use different measurement units, raw coefficient magnitudes should not be compared directly.

For example:

```text
annual_income → measured in dollars
fico_score → measured in points
years_employed → measured in years
```

---

## Standardized Coefficient Model

A separate model is trained using standardized features.

`StandardScaler` is fit only on the training dataset to prevent information leakage.

The standardized model makes coefficient magnitudes more comparable because variables are placed on a common scale.

This provides a clearer view of which variables have stronger associations with historical approval decisions.

---

# Model Evaluation

Model performance is evaluated exclusively on the held-out test dataset.

## Logistic Regression Accuracy

The model's classification accuracy is calculated on the test set.

The exact result is displayed in the notebook output.

---

## FICO Rule Baseline

The model is compared against Beacon's earlier rule:

```text
Approve if FICO >= 660
Reject otherwise
```

The baseline accuracy is calculated on the same test dataset.

This comparison helps determine whether considering multiple application characteristics provides additional predictive value beyond a single credit-score threshold.

---

## Always-Approve Baseline

A second baseline assumes that every application is approved.

```text
Prediction = Approved for every applicant
```

This establishes a simple reference point based on the majority class.

A useful model should provide value beyond this naive strategy.

---

# Probability-Based Triage

The trained model uses:

```python
predict_proba()
```

to calculate an approval probability for every test application.

Applications are then classified into operational confidence bands.

```text
p >= 0.80
→ Expedited workflow

0.20 <= p < 0.80
→ Send to loan officer

p < 0.20
→ Likely rejection, but mandatory human review
```

This approach focuses automation on high-confidence cases while directing uncertain applications to human reviewers.

---

# Why Logistic Regression?

Logistic regression was appropriate for this use case because the target is binary:

```text
Approved
Rejected
```

It also provides several benefits for a lending decision-support setting:

- Produces probabilities rather than only class labels
- Allows business-defined decision thresholds
- Is more interpretable than many complex black-box models
- Supports coefficient-based analysis
- Provides a strong benchmark for future models

The goal was not to maximize complexity, but to build a model that is understandable, reproducible, and useful for operational decision support.

---

# Fairness and Risk Considerations

The model is intentionally designed as a **human-in-the-loop system**.

Even if explicitly protected characteristics such as race or age are excluded, other variables may still correlate with protected characteristics.

Examples include:

- Income
- Savings
- Employment history
- Employment status

Historical lending decisions may also contain human inconsistency or bias.

Because the model learns from those historical decisions, it could reproduce those patterns.

For this reason:

- The model should not independently reject applicants
- Low-confidence and negative cases require human review
- Final responsibility remains with Beacon's loan officers
- Predictions should be periodically monitored for unfair outcomes
- Model behavior should be reassessed as applicant populations and lending policies change

---

# Limitations

The model has several important limitations.

### Historical Decisions, Not Credit Risk

The model predicts Beacon's historical approval decisions.

It does **not** estimate:

- Probability of default
- Probability of repayment
- Expected credit loss
- Long-term borrower performance

---

### Historical Bias

If historical officer decisions contain systematic inconsistencies or biases, the model may learn those patterns.

---

### Dataset Size

A relatively small historical dataset can make model performance sensitive to individual records and sampling variation.

Performance may change when applied to a larger or different applicant population.

---

### Changing Lending Policies

The model reflects historical decision patterns.

If Beacon changes its lending standards, risk policies, products, or community priorities, the model may require retraining.

---

### Correlation Is Not Causation

Model coefficients indicate relationships between applicant characteristics and historical decisions.

They should not be interpreted as proof that a particular characteristic caused an approval or rejection.

---

# Appropriate Use

The model is intended to support Beacon's loan officers by:

- Prioritizing applications
- Identifying high-confidence cases
- Reducing repetitive manual review
- Improving consistency
- Providing a reproducible decision-support framework
- Helping officers focus attention on ambiguous applications

It should **not** be used as an automatic rejection system.

Final lending decisions remain the responsibility of human loan officers.

---

# Key Takeaway

This solution demonstrates how an interpretable machine learning model can support a high-volume lending workflow without removing human accountability.

Instead of relying on a single FICO threshold, the model evaluates several characteristics together and produces an approval probability that can be used to prioritize applications.

The result is a more scalable decision-support process that balances:

**Efficiency + Consistency + Explainability + Human Oversight**
