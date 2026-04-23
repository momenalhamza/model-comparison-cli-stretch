"""
Module 5 Week B — Integration Task: Model Comparison & Decision Memo

Module 5 culminating deliverable. Compare 6 model configurations using
5-fold stratified cross-validation, produce PR curves and calibration
plots, log experiments, persist the best model, and demonstrate what
tree-based models capture that linear models cannot.

Complete the 9 functions below. See the integration guide for task-by-task
detail.
Run with:  python model_comparison.py
Tests:     pytest tests/ -v
"""

import os
from datetime import datetime

# Use a non-interactive matplotlib backend so plots save cleanly in CI
# and on headless environments.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.calibration import CalibrationDisplay
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (PrecisionRecallDisplay, average_precision_score,
                             make_scorer, precision_score, recall_score,
                             f1_score, accuracy_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges",
                    "num_support_calls", "senior_citizen",
                    "has_partner", "has_dependents", "contract_months"]


def load_and_preprocess(filepath="data/telecom_churn.csv", random_state=42):
    """Load the Petra Telecom dataset and split into train/test sets.

    Uses an 80/20 stratified split. Features are the 8 NUMERIC_FEATURES
    columns. Target is `churned`.

    Args:
        filepath: Path to telecom_churn.csv.
        random_state: Random seed for reproducible split.

    Returns:
        Tuple (X_train, X_test, y_train, y_test) where X contains only
        NUMERIC_FEATURES and y is the `churned` column.
    """
    df = pd.read_csv(filepath)

    X = df[NUMERIC_FEATURES]
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test




def define_models():
    """Define 6 model configurations for comparison.

    The pattern is deliberate: a default vs class_weight='balanced' pair
    at BOTH the linear and ensemble family levels. This lets you observe
    the class_weight effect at two levels of model complexity.

    The 6 configurations:
      1. DummyClassifier(strategy='most_frequent') — baseline
      2. LogisticRegression(max_iter=1000) — linear default (needs scaling)
      3. LogisticRegression(class_weight='balanced', max_iter=1000) — linear balanced (needs scaling)
      4. DecisionTreeClassifier(max_depth=5) — tree baseline
      5. RandomForestClassifier(n_estimators=100, max_depth=10) — ensemble default
      6. RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced') — ensemble balanced

    LR variants require StandardScaler preprocessing; tree-based models
    do not. Use sklearn Pipeline to pair each model with its preprocessing.

    Returns:
        Dict of {name: sklearn.pipeline.Pipeline} with 6 entries.
        Names: 'Dummy', 'LR_default', 'LR_balanced', 'DT_depth5',
               'RF_default', 'RF_balanced'.
    """
    models = {
        "Dummy": Pipeline([
            ("scaler", "passthrough"),
            ("model", DummyClassifier(strategy="most_frequent"))
        ]),

        "LR_default": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42))
        ]),

        "LR_balanced": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            ))
        ]),

        "DT_depth5": Pipeline([
            ("scaler", "passthrough"),
            ("model", DecisionTreeClassifier(max_depth=5, random_state=42))
        ]),

        "RF_default": Pipeline([
            ("scaler", "passthrough"),
            ("model", RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ))
        ]),

        "RF_balanced": Pipeline([
            ("scaler", "passthrough"),
            ("model", RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight="balanced",
                random_state=42
            ))
        ]),
    }

    return models


def run_cv_comparison(models, X, y, n_splits=5, random_state=42):
    """Run 5-fold stratified cross-validation on all models.

    For each model, compute mean and std of: accuracy, precision, recall,
    F1, and PR-AUC across folds. PR-AUC uses predict_proba — it is a
    threshold-independent ranking metric.

    Args:
        models: Dict of {name: Pipeline} from define_models().
        X: Feature DataFrame.
        y: Target Series.
        n_splits: Number of CV folds.
        random_state: Random seed for StratifiedKFold.

    Returns:
        DataFrame with columns: model, accuracy_mean, accuracy_std,
        precision_mean, precision_std, recall_mean, recall_std,
        f1_mean, f1_std, pr_auc_mean, pr_auc_std.
        One row per model (6 rows total).
    """
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    results = []

    for name, pipeline in models.items():
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        pr_auc_scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]

            pipeline.fit(X_train_fold, y_train_fold)

            y_pred = pipeline.predict(X_val_fold)
            y_proba = pipeline.predict_proba(X_val_fold)[:, 1]

            accuracy_scores.append(accuracy_score(y_val_fold, y_pred))
            precision_scores.append(
                precision_score(y_val_fold, y_pred, zero_division=0)
            )
            recall_scores.append(
                recall_score(y_val_fold, y_pred, zero_division=0)
            )
            f1_scores.append(
                f1_score(y_val_fold, y_pred, zero_division=0)
            )
            pr_auc_scores.append(
                average_precision_score(y_val_fold, y_proba)
            )

        results.append({
            "model": name,
            "accuracy_mean": np.mean(accuracy_scores),
            "accuracy_std": np.std(accuracy_scores),
            "precision_mean": np.mean(precision_scores),
            "precision_std": np.std(precision_scores),
            "recall_mean": np.mean(recall_scores),
            "recall_std": np.std(recall_scores),
            "f1_mean": np.mean(f1_scores),
            "f1_std": np.std(f1_scores),
            "pr_auc_mean": np.mean(pr_auc_scores),
            "pr_auc_std": np.std(pr_auc_scores),
        })

    return pd.DataFrame(results)


def save_comparison_table(results_df, output_path="results/comparison_table.csv"):
    """Save the comparison table to CSV.

    Args:
        results_df: DataFrame from run_cv_comparison().
        output_path: Destination path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)


def plot_pr_curves_top3(models, X_test, y_test, output_path="results/pr_curves.png"):
    """Plot PR curves for the top 3 models (by PR-AUC) on one axes and save.

    Args:
        models: Dict of {name: fitted Pipeline} — must already be fitted.
        X_test: Test features.
        y_test: Test labels.
        output_path: Destination path for the PNG.
    """
    pr_auc_scores = {}

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        pr_auc_scores[name] = average_precision_score(y_test, y_proba)

    top3 = sorted(pr_auc_scores.items(), key=lambda x: x[1], reverse=True)[:3]

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, _ in top3:
        PrecisionRecallDisplay.from_estimator(
            models[name],
            X_test,
            y_test,
            ax=ax,
            name=name
        )

    ax.set_title("Precision-Recall Curves for Top 3 Models")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

def plot_calibration_top3(models, X_test, y_test, output_path="results/calibration.png"):
    """Plot calibration curves for the top 3 models and save.

    Uses CalibrationDisplay.from_estimator.

    Args:
        models: Dict of {name: fitted Pipeline} — must already be fitted.
        X_test: Test features.
        y_test: Test labels.
        output_path: Destination path for the PNG.
    """
    pr_auc_scores = {}

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        pr_auc_scores[name] = average_precision_score(y_test, y_proba)

    top3 = sorted(pr_auc_scores.items(), key=lambda x: x[1], reverse=True)[:3]

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, _ in top3:
        CalibrationDisplay.from_estimator(
            models[name],
            X_test,
            y_test,
            n_bins=10,
            ax=ax,
            name=name
        )

    ax.set_title("Calibration Curves for Top 3 Models")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_best_model(best_model, output_path="results/best_model.joblib"):
    """Persist the best model to disk with joblib.

    Args:
        best_model: A fitted sklearn Pipeline.
        output_path: Destination path.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    dump(best_model, output_path)


def log_experiment(results_df, output_path="results/experiment_log.csv"):
    """Log all model results with timestamps.

    Produces a CSV with columns: model_name, accuracy, precision, recall,
    f1, pr_auc, timestamp. One row per model. The timestamp records WHEN
    the experiment was run (ISO format).

    Args:
        results_df: DataFrame from run_cv_comparison().
        output_path: Destination path.
    """
    timestamp = datetime.now().isoformat()

    log_df = pd.DataFrame({
        "model_name": results_df["model"],
        "accuracy": results_df["accuracy_mean"],
        "precision": results_df["precision_mean"],
        "recall": results_df["recall_mean"],
        "f1": results_df["f1_mean"],
        "pr_auc": results_df["pr_auc_mean"],
        "timestamp": timestamp
    })

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    log_df.to_csv(output_path, index=False)


def find_tree_vs_linear_disagreement(rf_model, lr_model, X_test, y_test,
                                     feature_names, min_diff=0.15):
    """Find ONE test sample where RF and LR predicted probabilities differ most."""
    rf_proba_all = rf_model.predict_proba(X_test)[:, 1]
    lr_proba_all = lr_model.predict_proba(X_test)[:, 1]

    prob_diffs = np.abs(rf_proba_all - lr_proba_all)

    max_pos = int(np.argmax(prob_diffs))
    max_diff = float(prob_diffs[max_pos])

    if max_diff < min_diff:
        return None

    sample_idx = int(X_test.index[max_pos])
    sample_row = X_test.iloc[max_pos]

    feature_values = {
        feature: sample_row[feature].item() if hasattr(sample_row[feature], "item") else sample_row[feature]
        for feature in feature_names
    }

    true_label = y_test.iloc[max_pos]
    true_label = int(true_label.item() if hasattr(true_label, "item") else true_label)

    return {
        "sample_idx": sample_idx,
        "feature_values": feature_values,
        "rf_proba": float(rf_proba_all[max_pos]),
        "lr_proba": float(lr_proba_all[max_pos]),
        "prob_diff": max_diff,
        "true_label": true_label
    }



def sweep_thresholds_and_recommend(
    model,
    X_test,
    y_test,
    thresholds=None,
    max_contacts=150,
    customer_base=10000,
    output_csv="results/threshold_metrics.csv"
):
    """Tier 1: evaluate thresholds and choose the best feasible one."""
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.05)

    y_proba = model.predict_proba(X_test)[:, 1]
    rows = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        alert_rate = float(y_pred.mean())

        rows.append({
            "threshold": round(float(threshold), 2),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "alerts_per_1000": alert_rate * 1000,
            "expected_alerts_per_10000": alert_rate * customer_base,
        })

    threshold_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    threshold_df.to_csv(output_csv, index=False)

    feasible = threshold_df[
        threshold_df["expected_alerts_per_10000"] <= max_contacts
    ].copy()

    recommendation = None
    if not feasible.empty:
        feasible = feasible.sort_values(
            ["recall", "f1", "expected_alerts_per_10000", "threshold"],
            ascending=[False, False, False, False]
        )
        recommendation = feasible.iloc[0].to_dict()

    return threshold_df, recommendation


def plot_threshold_sweep(
    threshold_df,
    recommendation=None,
    output_path="results/threshold_sweep.png"
):
    """Plot precision, recall, F1, and alerts per 1,000 vs threshold."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(
        threshold_df["threshold"],
        threshold_df["precision"],
        marker="o",
        label="Precision"
    )
    ax1.plot(
        threshold_df["threshold"],
        threshold_df["recall"],
        marker="o",
        label="Recall"
    )
    ax1.plot(
        threshold_df["threshold"],
        threshold_df["f1"],
        marker="o",
        label="F1"
    )

    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(
        threshold_df["threshold"],
        threshold_df["alerts_per_1000"],
        marker="s",
        linestyle="--",
        label="Alerts per 1,000"
    )
    ax2.axhline(
        15,
        linestyle=":",
        label="Capacity (15 alerts / 1,000)"
    )
    ax2.set_ylabel("Alerts per 1,000 customers")

    if recommendation is not None:
        ax1.axvline(
            recommendation["threshold"],
            linestyle="--"
        )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")

    ax1.set_title("Threshold Sweep for Recommended Model")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_threshold_recommendation(
    recommendation,
    model_name,
    output_path="results/threshold_recommendation.md"
):
    """Write a memo-ready threshold recommendation section."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if recommendation is None:
        text = (
            "# Threshold Recommendation\n\n"
            "No tested threshold satisfied the capacity constraint of "
            "150 contacts per 10,000 customers. Consider testing higher "
            "thresholds or revisiting the outreach limit.\n"
        )
    else:
        text = (
            "# Threshold Recommendation\n\n"
            f"For **{model_name}**, the recommended operating threshold is "
            f"**{recommendation['threshold']:.2f}**. "
            f"At this threshold, the model is expected to generate about "
            f"**{recommendation['alerts_per_1000']:.1f} alerts per 1,000 customers** "
            f"(approximately **{recommendation['expected_alerts_per_10000']:.0f} alerts "
            f"per 10,000 customers**), which fits Petra Telecom's monthly outreach "
            f"capacity of **150 contacts**. Under this constraint, this threshold gives "
            f"the **highest recall** among the tested options while keeping outreach "
            f"volume operationally feasible. The business trade-off is that lowering "
            f"the threshold would catch more churners but would exceed team capacity, "
            f"while raising it further would reduce wasted effort but miss more "
            f"at-risk customers.\n"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


def compute_permutation_importance_top3(
    fitted_models,
    results_df,
    X_test,
    y_test,
    n_repeats=10,
    output_csv="results/permutation_importance.csv"
):
    """Tier 2: permutation importance for top 3 models by PR-AUC."""
    top3_names = (
        results_df.sort_values("pr_auc_mean", ascending=False)
        .head(3)["model"]
        .tolist()
    )

    rows = []

    for name in top3_names:
        result = permutation_importance(
            fitted_models[name],
            X_test,
            y_test,
            n_repeats=n_repeats,
            random_state=42,
            scoring="average_precision"
        )

        for feature, mean_val, std_val in zip(
            NUMERIC_FEATURES,
            result.importances_mean,
            result.importances_std
        ):
            rows.append({
                "model": name,
                "feature": feature,
                "importance_mean": mean_val,
                "importance_std": std_val,
            })

    importance_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    importance_df.to_csv(output_csv, index=False)

    return importance_df, top3_names


def plot_permutation_importance_comparison(
    importance_df,
    model_names,
    output_path="results/permutation_importance.png"
):
    """Grouped bar chart for top 8 features across top 3 models."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    top_features = (
        importance_df.groupby("feature")["importance_mean"]
        .mean()
        .sort_values(ascending=False)
        .head(8)
        .index.tolist()
    )

    plot_df = importance_df[
        importance_df["feature"].isin(top_features)
    ].copy()

    pivot_df = plot_df.pivot(
        index="feature",
        columns="model",
        values="importance_mean"
    )

    pivot_df = pivot_df.reindex(top_features)
    pivot_df = pivot_df[[m for m in model_names if m in pivot_df.columns]]

    ax = pivot_df.plot(kind="bar", figsize=(11, 6))
    ax.set_title("Permutation Importance Comparison Across Top 3 Models")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Mean importance drop (Average Precision)")
    ax.legend(title="Model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

from sklearn.inspection import permutation_importance
def save_permutation_importance_summary(
    importance_df,
    output_path="results/permutation_importance_summary.md"
):
    """Write the interpretation paragraph for Tier 2."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    model_to_top = {}
    for model, grp in importance_df.groupby("model"):
        top_features = (
            grp.sort_values("importance_mean", ascending=False)["feature"]
            .head(5)
            .tolist()
        )
        model_to_top[model] = top_features

    model_names = list(model_to_top.keys())

    lr_name = next((m for m in model_names if m.startswith("LR")), None)
    rf_name = next((m for m in model_names if m.startswith("RF")), None)

    if lr_name and rf_name:
        model_a, model_b = lr_name, rf_name
    elif len(model_names) >= 2:
        model_a, model_b = model_names[0], model_names[1]
    else:
        model_a, model_b = None, None

    if model_a and model_b:
        overlap = sorted(set(model_to_top[model_a]) & set(model_to_top[model_b]))
        only_a = [f for f in model_to_top[model_a] if f not in overlap]
        only_b = [f for f in model_to_top[model_b] if f not in overlap]

        overlap_text = ", ".join(overlap[:3]) if overlap else "limited overlap"
        only_a_text = ", ".join(only_a[:3]) if only_a else "shared predictors"
        only_b_text = ", ".join(only_b[:3]) if only_b else "shared predictors"

        paragraph = (
            "# Permutation Importance Interpretation\n\n"
            f"Comparing **{model_a}** and **{model_b}**, the models agree most "
            f"clearly on **{overlap_text}** as important signals. At the same time, "
            f"**{model_a}** gives relatively more weight to **{only_a_text}**, while "
            f"**{model_b}** emphasizes **{only_b_text}**. This suggests the model "
            f"families are not using exactly the same decision process: linear models "
            f"usually reward broad additive trends, while tree-based models can place "
            f"more value on interactions, split points, and threshold-style effects.\n"
        )
    else:
        paragraph = (
            "# Permutation Importance Interpretation\n\n"
            "The feature rankings differ across the evaluated models, which suggests "
            "that each model family is relying on a somewhat different decision process "
            "even when headline performance is similar.\n"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(paragraph)


def main():
    """Orchestrate all 9 integration tasks. Run with: python model_comparison.py"""
    os.makedirs("results", exist_ok=True)

    # Task 1: Load + split
    result = load_and_preprocess()
    if not result:
        print("load_and_preprocess not implemented. Exiting.")
        return
    X_train, X_test, y_train, y_test = result
    print(f"Data: {len(X_train)} train, {len(X_test)} test, "
          f"churn rate: {y_train.mean():.2%}")

    # Task 2: Define models
    models = define_models()
    if not models:
        print("define_models not implemented. Exiting.")
        return
    print(f"\n{len(models)} model configurations defined: {list(models.keys())}")

    # Task 3: Cross-validation comparison
    results_df = run_cv_comparison(models, X_train, y_train)
    if results_df is None:
        print("run_cv_comparison not implemented. Exiting.")
        return
    print("\n=== Model Comparison Table (5-fold CV) ===")
    print(results_df.to_string(index=False))

    # Task 4: Save comparison table
    save_comparison_table(results_df)

    # Fit all models on full training set for plots + persistence
    fitted_models = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline

    # Task 5: PR curves (top 3)
    plot_pr_curves_top3(fitted_models, X_test, y_test)

    # Task 6: Calibration plot (top 3)
    plot_calibration_top3(fitted_models, X_test, y_test)

    # Task 7: Save best model
    best_name = results_df.sort_values("pr_auc_mean", ascending=False).iloc[0]["model"]
    print(f"\nBest model by PR-AUC: {best_name}")
    save_best_model(fitted_models[best_name])

    # Task 8: Experiment log
    log_experiment(results_df)


    # Challenge Tier 1: Threshold optimization for deployment
    threshold_df, threshold_recommendation = sweep_thresholds_and_recommend(
        fitted_models[best_name],
        X_test,
        y_test
    )
    plot_threshold_sweep(threshold_df, threshold_recommendation)
    save_threshold_recommendation(threshold_recommendation, best_name)

    if threshold_recommendation is not None:
        print(
            f"\nThreshold recommendation for {best_name}: "
            f"{threshold_recommendation['threshold']:.2f} "
            f"(expected alerts per 10,000 = "
            f"{threshold_recommendation['expected_alerts_per_10000']:.0f})"
        )

    # Challenge Tier 2: Permutation importance
    importance_df, top3_names = compute_permutation_importance_top3(
        fitted_models,
        results_df,
        X_test,
        y_test
    )
    plot_permutation_importance_comparison(importance_df, top3_names)
    save_permutation_importance_summary(importance_df)


    # Task 9: Tree-vs-linear disagreement
    rf_pipeline = fitted_models["RF_default"]
    lr_pipeline = fitted_models["LR_default"]
    disagreement = find_tree_vs_linear_disagreement(
        rf_pipeline, lr_pipeline, X_test, y_test, NUMERIC_FEATURES
    )
    if disagreement:
        print(f"\n--- Tree-vs-linear disagreement (sample idx={disagreement['sample_idx']}) ---")
        print(f"  RF P(churn=1)={disagreement['rf_proba']:.3f}  "
              f"LR P(churn=1)={disagreement['lr_proba']:.3f}")
        print(f"  |diff| = {disagreement['prob_diff']:.3f}   "
              f"true label = {disagreement['true_label']}")

        # Save disagreement analysis to markdown
        md_lines = [
            "# Tree vs. Linear Disagreement Analysis",
            "",
            "## Sample Details",
            "",
            f"- **Test-set index:** {disagreement['sample_idx']}",
            f"- **True label:** {disagreement['true_label']}",
            f"- **RF predicted P(churn=1):** {disagreement['rf_proba']:.4f}",
            f"- **LR predicted P(churn=1):** {disagreement['lr_proba']:.4f}",
            f"- **Probability difference:** {disagreement['prob_diff']:.4f}",
            "",
            "## Feature Values",
            "",
        ]
        for feat, val in disagreement["feature_values"].items():
            md_lines.append(f"- **{feat}:** {val}")
        md_lines.extend([
            "",
            "## Structural Explanation",
"The random forest likely reacted strongly to a threshold-style pattern around `contract_months = 1`, where short contracts can sharply increase churn risk when combined with other customer attributes such as no partner and no dependents. Logistic regression, by contrast, combines features additively, so the low monthly charges and moderate tenure pulled the predicted probability downward instead of allowing a sharp rule-like jump. This illustrates how the tree model can capture feature interactions and threshold effects that a linear model cannot represent as naturally."
            
        ])
        with open("results/tree_vs_linear_disagreement.md", "w") as f:
            f.write("\n".join(md_lines))
        print("  Saved to results/tree_vs_linear_disagreement.md")

    print("\n--- All results saved to results/ ---")
    print("Write your decision memo in the PR description (Task 10).")


if __name__ == "__main__":
    main()
