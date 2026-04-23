# Permutation Importance Interpretation

Comparing **LR_default** and **RF_balanced**, the models agree most clearly on **contract_months, num_support_calls, tenure** as important signals. At the same time, **LR_default** gives relatively more weight to **has_dependents**, while **RF_balanced** emphasizes **monthly_charges**. This suggests the model families are not using exactly the same decision process: linear models usually reward broad additive trends, while tree-based models can place more value on interactions, split points, and threshold-style effects.
