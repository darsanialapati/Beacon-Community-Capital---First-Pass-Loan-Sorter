# Beacon-Community-Capital---First-Pass-Loan-Sorter

### Problem

Beacon Community Capital was experiencing a significant increase in loan applications, growing from roughly 500 to 1,800 applications per month, while the lending team remained limited to five loan officers. Because every application was reviewed manually, decision times increased to three to four business days.

The manual process also introduced inconsistency. Historical reviews showed that different officers could arrive at different decisions for the same application, and the reasoning behind approvals or rejections was not always documented in a consistent way. Beacon had previously tried using a simple FICO cutoff of 660, but this approach was too restrictive because it ignored other important factors such as income, savings, employment history, and loan characteristics.

The historical application data also contained quality issues such as missing values, duplicate records, invalid values, and numeric fields stored incorrectly, making data preparation an important part of the solution.

### Objective

I built a machine learning solution to support Beacon's loan officers by providing a consistent first-pass assessment of incoming applications.

The goal was not to predict whether a borrower would default or repay a loan. Instead, the model was designed to learn from Beacon's historical lending decisions and estimate how likely an application was to have been approved by the credit team based on the information available at the time.

The solution was intended to reduce the amount of manual review required for straightforward applications while ensuring that uncertain or potentially sensitive cases continued to receive human review.

### Solution

I cleaned and prepared Beacon's historical loan application data and developed a logistic regression classification model using application-level factors such as credit score, income, loan amount, employment history, savings, employment status, and other available borrower characteristics.

The model produces an approval probability between 0 and 1 for every application. Rather than automatically approving or rejecting applicants, I converted these probabilities into an operational triage framework.

Applications with a predicted approval probability of 80% or higher are identified as strong candidates for an expedited approval workflow. Applications between 20% and 80% are treated as borderline cases and routed to a loan officer for detailed review. Applications below 20% are flagged as likely rejections but still require a human officer to review and make the final decision.

This approach provides Beacon with a more scalable and consistent way to prioritize applications while preserving human accountability. It also improves transparency because every prediction is based on defined application characteristics rather than a single credit-score rule, allowing the model to consider the broader financial profile of each applicant.
