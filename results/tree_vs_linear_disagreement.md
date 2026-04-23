# Tree vs. Linear Disagreement Analysis

## Sample Details

- **Test-set index:** 4060
- **True label:** 0
- **RF predicted P(churn=1):** 0.5998
- **LR predicted P(churn=1):** 0.1700
- **Probability difference:** 0.4299

## Feature Values

- **tenure:** 36.0
- **monthly_charges:** 20.0
- **total_charges:** 1077.33
- **num_support_calls:** 2.0
- **senior_citizen:** 0.0
- **has_partner:** 0.0
- **has_dependents:** 0.0
- **contract_months:** 1.0

## Structural Explanation
The random forest likely reacted strongly to a threshold-style pattern around `contract_months = 1`, where short contracts can sharply increase churn risk when combined with other customer attributes such as no partner and no dependents. Logistic regression, by contrast, combines features additively, so the low monthly charges and moderate tenure pulled the predicted probability downward instead of allowing a sharp rule-like jump. This illustrates how the tree model can capture feature interactions and threshold effects that a linear model cannot represent as naturally.