# Stretch: Production-Ready Model Comparison CLI

A command-line interface for training, evaluating, and comparing machine learning models on classification tasks.

## Description

`stretch.py` is a production-quality CLI tool that compares multiple classifiers using stratified cross-validation, generates visualizations (PR curves, calibration plots), and persists the best-performing model.

**Supported Models:**
- Logistic Regression (default, balanced)
- SVM (default, balanced)
- Decision Tree (max_depth=5)
- Random Forest (default, balanced)
- XGBoost (default, balanced) - optional

## Installation & Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- scikit-learn>=1.5,<1.10
- pandas
- numpy
- matplotlib
- joblib

**Optional for XGBoost support:**
```bash
pip install xgboost
```

## Usage

```bash
python stretch.py [OPTIONS]
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data-path` | No | data/telecom_churn.csv | Path to the input CSV dataset |
| `--output-dir` | No | ./output | Directory for results and plots |
| `--n-folds` | No | 5 | Number of CV folds (cross-validation) |
| `--random-seed` | No | 42 | Random seed for reproducibility |
| `--dry-run` | No | False | Validate data without training |
| `--debug` | No | False | Enable debug logging |

### Required Data Format

The input CSV must contain:
- Numeric feature columns: `tenure`, `monthly_charges`, `total_charges`, `num_support_calls`, `senior_citizen`, `has_partner`, `has_dependents`, `contract_months`
- Target column: `churned` (binary 0/1)

## Examples

### Example 1: Normal Run (Default)

Train all models with default settings:

```bash
python stretch.py
```

Output saved to `./output/`:
- `comparison_table.csv` - CV metrics for all models
- `pr_curves.png` - Precision-Recall curves (top 3)
- `calibration.png` - Calibration curves (top 3)
- `best_model.joblib` - Serialized best model
- `experiment_log.csv` - Timestamped results

### Example 2: Dry-Run Validation

Validate data and configuration without training:

```bash
python stretch.py --dry-run
```

Output shows:
- Data file validation (rows, columns, missing values)
- Target class distribution
- Pipeline configuration summary
- Command to run training

### Example 3: Full Customization

10-fold CV with custom output directory and seed:

```bash
python stretch.py \
    --output-dir ./results \
    --n-folds 10 \
    --random-seed 123 \
    --debug
```

### Example 4: Custom Data Path

```bash
python stretch.py --data-path /path/to/your/data.csv
```

## Testing

```bash
# View help
python stretch.py --help

# Validate data without training
python stretch.py --dry-run

# Full training with custom settings
python stretch.py --output-dir ./results --n-folds 5
```

## Output Structure

```
output/
├── comparison_table.csv    # Model performance metrics
├── pr_curves.png          # Precision-Recall curves
├── calibration.png        # Calibration curves
├── best_model.joblib      # Best model (by PR-AUC)
└── experiment_log.csv     # Timestamped experiment log
```

## Logging

- INFO: Progress updates, data loading, model training
- WARNING: XGBoost not installed, etc.
- DEBUG: Detailed per-fold metrics (use `--debug`)

## Exit Codes

- `0` - Success
- `1` - Data validation failure or missing file

## License

See [LICENSE](LICENSE) for terms.
