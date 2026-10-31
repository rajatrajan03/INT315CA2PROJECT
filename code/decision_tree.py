"""Simple Decision Tree Regression model for student interaction activity.

This module is separate from Spark graph-analysis scripts.
It reads processed student activity data, trains a Decision Tree regressor,
and saves metrics and predictions in the output folder.
"""

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
STUDENT_ACTIVITY_PATH = OUTPUT_DIR / "student_activity.csv"
METRICS_PATH = OUTPUT_DIR / "decision_tree_metrics.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "decision_tree_predictions.csv"
PLOT_DIR = OUTPUT_DIR / "decision_tree_plots"
SITE_ASSET_DIR = PROJECT_DIR / "showcase_site" / "assets"


def load_student_activity_data():
    """Load student-level activity data prepared by the project pipeline."""
    if not STUDENT_ACTIVITY_PATH.exists():
        raise FileNotFoundError(
            f"Required file not found: {STUDENT_ACTIVITY_PATH}. "
            "Please run the project data pipeline first."
        )

    return pd.read_csv(STUDENT_ACTIVITY_PATH)


def prepare_features_and_target(df):
    """Prepare model features and target with basic missing-value handling.

    Target:
    - total_weight

    Features:
    - interaction_count
    - network (one-hot encoded)

    This avoids data leakage because total_weight is only used as target.
    """
    working_df = df.copy()
    working_df = working_df.dropna(subset=["total_weight", "interaction_count", "network"])

    X = pd.get_dummies(
        working_df[["interaction_count", "network"]],
        columns=["network"],
        drop_first=True,
    )
    y = working_df["total_weight"]

    return X, y, working_df


def save_plot(fig, file_name):
    """Save a plot in the output folder and in the site assets folder."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    SITE_ASSET_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PLOT_DIR / file_name
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    shutil.copy2(output_path, SITE_ASSET_DIR / file_name)
    plt.close(fig)


def save_model_visuals(model, X_test, y_test, predictions):
    """Create plots that explain how the tree-based model behaves."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, predictions, alpha=0.8, color="#1f5f99")
    min_value = min(y_test.min(), predictions.min())
    max_value = max(y_test.max(), predictions.max())
    ax.plot([min_value, max_value], [min_value, max_value], "r--", linewidth=1.5)
    ax.set_xlabel("Actual total_weight")
    ax.set_ylabel("Predicted total_weight")
    ax.set_title("Decision Tree: Actual vs Predicted")
    save_plot(fig, "decision_tree_actual_vs_predicted.png")

    residuals = y_test.to_numpy() - predictions
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(predictions, residuals, alpha=0.8, color="#1d7a6f")
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Predicted total_weight")
    ax.set_ylabel("Residual")
    ax.set_title("Decision Tree: Residual Plot")
    save_plot(fig, "decision_tree_residuals.png")

    importances = model.feature_importances_
    feature_names = X_test.columns.tolist()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(feature_names, importances, color="#8b5cf6")
    ax.set_title("Decision Tree: Feature Importance")
    ax.set_ylabel("Importance")
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    save_plot(fig, "decision_tree_feature_importance.png")


def train_and_evaluate_model(X, y):
    """Train DecisionTreeRegressor and compute evaluation metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = DecisionTreeRegressor(random_state=42, max_depth=5)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    return model, X_test, y_test, predictions, {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def save_outputs(metrics_dict, results_df):
    """Save model metrics and test predictions to CSV files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(
        [{
            "model": "DecisionTreeRegressor",
            "MAE": metrics_dict["MAE"],
            "RMSE": metrics_dict["RMSE"],
            "R2": metrics_dict["R2"],
        }]
    )
    metrics_df.to_csv(METRICS_PATH, index=False)

    predictions_df = results_df[[
        "network",
        "student_id",
        "label",
        "actual_total_weight",
        "predicted_total_weight",
    ]].copy()
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)

    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved predictions: {PREDICTIONS_PATH}")


def main():
    """Run the Decision Tree regression workflow."""
    print("Loading student activity data...")
    student_df = load_student_activity_data()

    print("Preparing features and target...")
    X, y, prepared_df = prepare_features_and_target(student_df)

    # Protect against tiny/invalid datasets.
    if len(X) < 5:
        print(
            "Warning: Dataset is too small for a meaningful Decision Tree model "
            f"(valid rows: {len(X)})."
        )
        return

    print("Training Decision Tree model...")
    model, X_test, y_test, predictions, metrics = train_and_evaluate_model(X, y)
    save_model_visuals(model, X_test, y_test, predictions)

    test_results = prepared_df.loc[X_test.index, ["network", "student_id", "label"]].copy().reset_index(drop=True)
    test_results["actual_total_weight"] = y_test.reset_index(drop=True).values
    test_results["predicted_total_weight"] = pd.Series(predictions).round(4)

    save_outputs(metrics, test_results)

    print("\nDecision Tree Evaluation")
    print("-" * 40)
    print(f"MAE:  {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"R2:   {metrics['R2']:.4f}")

    print("\nSample predictions:")
    print(test_results.head().to_string(index=False))

    print("\nDecision Tree model completed successfully.")


if __name__ == "__main__":
    main()
