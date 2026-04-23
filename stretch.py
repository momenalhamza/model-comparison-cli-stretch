"""
Stretch: Production-ready CLI tool for model comparison pipeline.

A command-line interface for training, evaluating, and comparing machine learning
models on classification tasks. Supports cross-validation, visualization,
and dry-run validation.

Example usage:
    python stretch.py --data-path data/telecom_churn.csv
    python stretch.py --data-path data.csv --output-dir ./results --n-folds 10 --dry-run
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.calibration import CalibrationDisplay
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (PrecisionRecallDisplay, accuracy_score,
                             average_precision_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


NUMERIC_FEATURES = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "num_support_calls",
    "senior_citizen",
    "has_partner",
    "has_dependents",
    "contract_months",
]
TARGET_COLUMN = "churned"


def setup_logging(debug: bool = False) -> logging.Logger:
    """Configure logging with appropriate level and format.

    Args:
        debug: If True, set logging level to DEBUG.

    Returns:
        Configured logger instance.
    """
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse and return command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="stretch",
        description="Production-ready CLI tool for model comparison pipeline. "
        "Trains and evaluates multiple classifiers using cross-validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with default settings
    python stretch.py --data-path data/telecom_churn.csv

    # Custom output directory and 10-fold cross-validation
    python stretch.py --data-path data.csv --output-dir ./results --n-folds 10

    # Dry-run mode: validate data without training
    python stretch.py --data-path data.csv --dry-run

    # Full customization with fixed random seed
    python stretch.py --data-path data.csv --output-dir ./results --n-folds 5 --random-seed 123
        """,
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/telecom_churn.csv",
        help="Path to the input CSV dataset (default: data/telecom_churn.csv).",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory for saving results and plots (default: ./output).",
    )

    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5).",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and configuration without training models.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for detailed diagnostics.",
    )

    return parser.parse_args()


def load_data(path: str) -> Optional[pd.DataFrame]:
    """Load data from CSV file.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with loaded data, or None if file not found.

    Raises:
        SystemExit: If file is not found or cannot be read.
    """
    logger = logging.getLogger(__name__)

    if not os.path.exists(path):
        logger.error(f"Data file not found: {path}")
        sys.exit(1)

    try:
        logger.info(f"Loading data from {path}")
        df = pd.read_csv(path)
        logger.info(
            f"Successfully loaded data: {len(df):,} rows, {len(df.columns)} columns"
        )
        logger.debug(
            f"Columns: {', '.join(df.columns.tolist())}"
        )
        return df
    except pd.errors.EmptyDataError:
        logger.error(f"Data file is empty: {path}")
        sys.exit(1)
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing data file {path}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load data from {path}: {e}")
        sys.exit(1)


def validate_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate that the data has required columns and structure.

    Args:
        df: Input DataFrame to validate.

    Returns:
        Tuple of (is_valid, list_of_issues).
    """
    logger = logging.getLogger(__name__)
    issues = []

    missing_features = [col for col in NUMERIC_FEATURES if col not in df.columns]
    if missing_features:
        issues.append(f"Missing required feature columns: {missing_features}")

    if TARGET_COLUMN not in df.columns:
        issues.append(f"Missing target column: '{TARGET_COLUMN}'")

    if len(df) == 0:
        issues.append("Dataset is empty (0 rows)")

    if issues:
        for issue in issues:
            logger.error(f"Validation issue: {issue}")
        return False, issues

    logger.info("Data validation passed")
    logger.debug(f"Target '{TARGET_COLUMN}' distribution:")
    value_counts = df[TARGET_COLUMN].value_counts().to_dict()
    for value, count in value_counts.items():
        pct = count / len(df) * 100
        logger.debug(f"  {value}: {count} ({pct:.1f}%)")

    return True, []


def print_configuration(args: argparse.Namespace, output_dir: str, logger: logging.Logger) -> None:
    """Print full pipeline configuration.

    Args:
        args: Parsed command-line arguments.
        output_dir: Absolute path to output directory.
        logger: Logger instance.
    """
    logger.info("=" * 60)
    logger.info("Pipeline Configuration")
    logger.info("=" * 60)
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Cross-validation: {args.n_folds}-fold stratified")
    logger.info(f"Random seed: {args.random_seed}")
    logger.info(f"Dry-run: {args.dry_run}")
    logger.info("-" * 60)
    logger.info("Models to compare:")
    logger.info(" 1. DummyClassifier (most_frequent) - baseline")
    logger.info(" 2. LogisticRegression (default) - linear")
    logger.info(" 3. LogisticRegression (balanced) - linear with class_weight")
    logger.info(" 4. SVC (default) - support vector machine")
    logger.info(" 5. SVC (balanced) - SVM with class_weight")
    logger.info(" 6. DecisionTreeClassifier (max_depth=5) - tree baseline")
    logger.info(" 7. RandomForestClassifier (default) - ensemble")
    logger.info(" 8. RandomForestClassifier (balanced) - ensemble with class_weight")
    logger.info(" 9. XGBClassifier (default) - gradient boosting (if available)")
    logger.info(" 10. XGBClassifier (balanced) - XGB with scale_pos_weight (if available)")
    logger.info("-" * 60)


def define_models(random_state: int, include_xgboost: bool = True) -> Dict[str, Pipeline]:
    """Define model configurations as sklearn Pipelines.

    Supports linear (Logistic Regression, SVM) and tree-based models
    (Decision Tree, Random Forest). XGBoost is included when available.

    Args:
        random_state: Random seed for reproducibility.
        include_xgboost: Whether to include XGBoost models if available.

    Returns:
        Dictionary mapping model names to Pipeline objects.
    """
    models = {
        # Dummy baseline
        "Dummy": Pipeline([
            ("scaler", "passthrough"),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]),

        # Linear models (Logistic Regression)
        "LR_default": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]),
        "LR_balanced": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=random_state,
            )),
        ]),

        # SVM models
        "SVM_default": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                probability=True,
                random_state=random_state,
            )),
        ]),
        "SVM_balanced": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                class_weight="balanced",
                probability=True,
                random_state=random_state,
            )),
        ]),

        # Tree baseline
        "DT_depth5": Pipeline([
            ("scaler", "passthrough"),
            ("model", DecisionTreeClassifier(max_depth=5, random_state=random_state)),
        ]),

        # Random Forest models
        "RF_default": Pipeline([
            ("scaler", "passthrough"),
            ("model", RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
            )),
        ]),
        "RF_balanced": Pipeline([
            ("scaler", "passthrough"),
            ("model", RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
    }

    # Add XGBoost if available
    if include_xgboost:
        try:
            from xgboost import XGBClassifier
            models["XGB_default"] = Pipeline([
                ("scaler", "passthrough"),
                ("model", XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    random_state=random_state,
                    use_label_encoder=False,
                    eval_metric="logloss",
                )),
            ])
            models["XGB_balanced"] = Pipeline([
                ("scaler", "passthrough"),
                ("model", XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    scale_pos_weight=5.0,  # Approximate for ~16% churn rate
                    random_state=random_state,
                    use_label_encoder=False,
                    eval_metric="logloss",
                )),
            ])
        except ImportError:
            logger = logging.getLogger(__name__)
            logger.warning(
                "XGBoost not installed. Skipping XGB models. "
                "Install with: pip install xgboost"
            )

    return models


def run_cv_comparison(
    models: Dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    random_state: int,
) -> pd.DataFrame:
    """Run stratified cross-validation on all models.

    Args:
        models: Dictionary of model name to Pipeline.
        X: Feature DataFrame.
        y: Target Series.
        n_splits: Number of CV folds.
        random_state: Random seed for StratifiedKFold.

    Returns:
        DataFrame with mean and std of all metrics for each model.
    """
    logger = logging.getLogger(__name__)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results = []

    for name, pipeline in models.items():
        logger.info(f"Training model: {name}")

        acc_scores = []
        prec_scores = []
        rec_scores = []
        f1_scores = []
        pr_auc_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            logger.debug(f" Fold {fold_idx}/{n_splits}")

            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]

            pipeline.fit(X_train_fold, y_train_fold)

            y_pred = pipeline.predict(X_val_fold)
            y_proba = pipeline.predict_proba(X_val_fold)[:, 1]

            acc_scores.append(accuracy_score(y_val_fold, y_pred))
            prec_scores.append(precision_score(y_val_fold, y_pred, zero_division=0))
            rec_scores.append(recall_score(y_val_fold, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_val_fold, y_pred, zero_division=0))
            pr_auc_scores.append(average_precision_score(y_val_fold, y_proba))

        results.append({
            "model": name,
            "accuracy_mean": np.mean(acc_scores),
            "accuracy_std": np.std(acc_scores),
            "precision_mean": np.mean(prec_scores),
            "precision_std": np.std(prec_scores),
            "recall_mean": np.mean(rec_scores),
            "recall_std": np.std(rec_scores),
            "f1_mean": np.mean(f1_scores),
            "f1_std": np.std(f1_scores),
            "pr_auc_mean": np.mean(pr_auc_scores),
            "pr_auc_std": np.std(pr_auc_scores),
        })

    results_df = pd.DataFrame(results)
    logger.info(f"Cross-validation complete for {len(models)} models")

    return results_df


def save_comparison_table(results_df: pd.DataFrame, output_path: str) -> None:
    """Save the comparison table to CSV.

    Args:
        results_df: DataFrame with model results.
        output_path: Path to save the CSV file.
    """
    logger = logging.getLogger(__name__)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"Saved comparison table to {output_path}")


def plot_pr_curves_top3(
    models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: str,
) -> None:
    """Plot precision-recall curves for top 3 models by PR-AUC.

    Args:
        models: Dictionary of fitted models.
        X_test: Test features.
        y_test: Test labels.
        output_path: Path to save the plot.
    """
    logger = logging.getLogger(__name__)

    pr_auc_scores = {}
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        pr_auc_scores[name] = average_precision_score(y_test, y_proba)

    top3 = sorted(pr_auc_scores.items(), key=lambda x: x[1], reverse=True)[:3]

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, _ in top3:
        PrecisionRecallDisplay.from_estimator(models[name], X_test, y_test, ax=ax, name=name)

    ax.set_title("Precision-Recall Curves for Top 3 Models")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    logger.info(f"Saved PR curves plot to {output_path}")


def plot_calibration_top3(
    models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: str,
) -> None:
    """Plot calibration curves for top 3 models.

    Args:
        models: Dictionary of fitted models.
        X_test: Test features.
        y_test: Test labels.
        output_path: Path to save the plot.
    """
    logger = logging.getLogger(__name__)

    pr_auc_scores = {}
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        pr_auc_scores[name] = average_precision_score(y_test, y_proba)

    top3 = sorted(pr_auc_scores.items(), key=lambda x: x[1], reverse=True)[:3]

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, _ in top3:
        CalibrationDisplay.from_estimator(
            models[name], X_test, y_test, n_bins=10, ax=ax, name=name
        )

    ax.set_title("Calibration Curves for Top 3 Models")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    logger.info(f"Saved calibration plot to {output_path}")


def save_best_model(results_df: pd.DataFrame, models: Dict[str, Pipeline], output_path: str) -> str:
    """Identify and save the best model by PR-AUC.

    Args:
        results_df: DataFrame with cross-validation results.
        models: Dictionary of fitted models.
        output_path: Path to save the best model.

    Returns:
        Name of the best model.
    """
    logger = logging.getLogger(__name__)

    best_name = results_df.sort_values("pr_auc_mean", ascending=False).iloc[0]["model"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dump(models[best_name], output_path)
    logger.info(f"Best model ({best_name}) saved to {output_path}")

    return best_name


def log_experiment(results_df: pd.DataFrame, output_path: str) -> None:
    """Log all model results with timestamps.

    Args:
        results_df: DataFrame with cross-validation results.
        output_path: Path to save the experiment log.
    """
    logger = logging.getLogger(__name__)

    timestamp = datetime.now().isoformat()

    log_df = pd.DataFrame({
        "model_name": results_df["model"],
        "accuracy": results_df["accuracy_mean"],
        "precision": results_df["precision_mean"],
        "recall": results_df["recall_mean"],
        "f1": results_df["f1_mean"],
        "pr_auc": results_df["pr_auc_mean"],
        "timestamp": timestamp,
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    log_df.to_csv(output_path, index=False)
    logger.info(f"Saved experiment log to {output_path}")


def train_and_evaluate(
    df: pd.DataFrame,
    output_dir: str,
    n_folds: int,
    random_seed: int,
) -> None:
    """Run the full model comparison pipeline.

    Args:
        df: Input DataFrame with features and target.
        output_dir: Directory to save outputs.
        n_folds: Number of cross-validation folds.
        random_seed: Random seed for reproducibility.
    """
    logger = logging.getLogger(__name__)

    os.makedirs(output_dir, exist_ok=True)

    X = df[NUMERIC_FEATURES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=y
    )

    logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test samples")
    logger.info(f"Training churn rate: {y_train.mean():.2%}")

    models = define_models(random_seed)

    results_df = run_cv_comparison(models, X_train, y_train, n_folds, random_seed)

    logger.info("\n" + "=" * 60)
    logger.info("Model Comparison Results (Cross-Validation)")
    logger.info("=" * 60)
    for _, row in results_df.iterrows():
        logger.info(f"{row['model']:15s}: "
                   f"PR-AUC={row['pr_auc_mean']:.4f} (+/- {row['pr_auc_std']:.4f}), "
                   f"F1={row['f1_mean']:.4f}, "
                   f"Recall={row['recall_mean']:.4f}")

    save_comparison_table(results_df, os.path.join(output_dir, "comparison_table.csv"))

    fitted_models = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline

    plot_pr_curves_top3(fitted_models, X_test, y_test, os.path.join(output_dir, "pr_curves.png"))

    plot_calibration_top3(fitted_models, X_test, y_test, os.path.join(output_dir, "calibration.png"))

    best_name = save_best_model(results_df, fitted_models, os.path.join(output_dir, "best_model.joblib"))

    log_experiment(results_df, os.path.join(output_dir, "experiment_log.csv"))

    logger.info(f"\nBest model by PR-AUC: {best_name}")
    best_row = results_df[results_df["model"] == best_name].iloc[0]
    logger.info(f" PR-AUC: {best_row['pr_auc_mean']:.4f} (+/- {best_row['pr_auc_std']:.4f})")

    logger.info(f"\nAll results saved to: {output_dir}")


def save_results(
    output_dir: str,
    results: Optional[Dict] = None,
) -> None:
    """Save results to output directory.

    This function creates a summary file in the output directory.

    Args:
        output_dir: Directory to save results.
        results: Optional dictionary of results to save.
    """
    logger = logging.getLogger(__name__)

    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Model Comparison Pipeline - {datetime.now().isoformat()}\n")
        f.write("=" * 60 + "\n")
        if results:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")

    logger.info(f"Saved results summary to {summary_path}")


def main() -> int:
    """Main entry point for the CLI.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    args = parse_arguments()
    logger = setup_logging(debug=args.debug)

    logger.info("Starting model comparison pipeline")

    df = load_data(args.data_path)
    if df is None:
        return 1

    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    is_valid, issues = validate_data(df)
    if not is_valid:
        logger.error("Data validation failed. Cannot proceed.")
        for issue in issues:
            logger.error(f" - {issue}")
        return 1

    logger.info("Data validation passed")

    class_dist = df[TARGET_COLUMN].value_counts().to_dict()
    logger.info(f"Target '{TARGET_COLUMN}' distribution: {class_dist}")

    output_dir = os.path.abspath(args.output_dir)
    print_configuration(args, output_dir, logger)

    if args.dry_run:
        logger.info("=" * 60)
        logger.info("DRY-RUN MODE: Validation complete, no training will be performed")
        logger.info("=" * 60)
        logger.info("Data file is valid and ready for training")
        logger.info(f"To run training, execute without --dry-run flag:")
        logger.info(f" python stretch.py --data-path {args.data_path} --output-dir {args.output_dir}")
        return 0

    logger.info("Starting model training and evaluation...")
    train_and_evaluate(df, output_dir, args.n_folds, args.random_seed)

    logger.info("Pipeline completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
