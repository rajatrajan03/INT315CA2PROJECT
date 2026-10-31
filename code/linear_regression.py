"""Simple Linear Regression model for student interaction activity.

This module is intentionally separate from the Spark graph-analysis code.
It reads the processed student activity dataset, trains a linear regression model,
and saves evaluation metrics and predictions to the output folder.
"""

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
STUDENT_ACTIVITY_PATH = OUTPUT_DIR / "student_activity.csv"
METRICS_PATH = OUTPUT_DIR / "linear_regression_metrics.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "linear_regression_predictions.csv"
PLOT_DIR = OUTPUT_DIR / "linear_regression_plots"
SITE_ASSET_DIR = PROJECT_DIR / "showcase_site" / "assets"


def load_student_activity_data():
    """Load the processed student activity dataset.

    We use this file because it already contains both the student-level activity
    summary and the target variable we want to predict: total_weight.
    """
    if not STUDENT_ACTIVITY_PATH.exists():
        raise FileNotFoundError(
            f"Required file not found: {STUDENT_ACTIVITY_PATH}. "
            "Please run the project data pipeline first."
        )

    df = pd.read_csv(STUDENT_ACTIVITY_PATH)

    # Keep the model simple and easy to understand.
    # We predict total interaction weight using interaction_count and the network type.
    # The target (total_weight) is not used as an input feature.
    return df


def prepare_features_and_target(df):
    """Prepare the regression dataset.

    This keeps the model beginner-friendly:
    - target: total_weight
    - features: interaction_count + one-hot encoded network

    We avoid leakage by ensuring total_weight is never used as a predictor.
    """
    working_df = df.copy()

    # Handle missing values safely.
    working_df = working_df.dropna(subset=["total_weight", "interaction_count"])

    # The network is categorical, so we convert it into dummy variables.
    features_df = pd.get_dummies(
        working_df[["network", "interaction_count"]],
        columns=["network"],
        drop_first=True,
    )

    target = working_df["total_weight"]

    return features_df, target, working_df


def save_plot(fig, file_name):
    """Save a plot in both the output folder and the site assets folder."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    SITE_ASSET_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PLOT_DIR / file_name
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    shutil.copy2(output_path, SITE_ASSET_DIR / file_name)
    plt.close(fig)


def save_model_visuals(model, X_test, y_test, predictions):
    """Create beginner-friendly visualizations for the model evaluation."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, predictions, alpha=0.8, color="#2d6fa5")
    min_value = min(y_test.min(), predictions.min())
    max_value = max(y_test.max(), predictions.max())
    ax.plot([min_value, max_value], [min_value, max_value], "r--", linewidth=1.5)
    ax.set_xlabel("Actual total_weight")
    ax.set_ylabel("Predicted total_weight")
    ax.set_title("Linear Regression: Actual vs Predicted")
    save_plot(fig, "linear_regression_actual_vs_predicted.png")

    residuals = y_test.to_numpy() - predictions
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(predictions, residuals, alpha=0.8, color="#4c8d7c")
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Predicted total_weight")
    ax.set_ylabel("Residual")
    ax.set_title("Linear Regression: Residual Plot")
    save_plot(fig, "linear_regression_residuals.png")

    features = X_test.columns.tolist()
    coefficients = model.coef_
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(features, coefficients, color="#d97706")
    ax.set_title("Linear Regression: Feature Coefficients")
    ax.set_ylabel("Coefficient value")
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha="right")
    save_plot(fig, "linear_regression_coefficients.png")


def train_and_evaluate_model(X, y):
    """Train a LinearRegression model and return metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    return model, X_train, X_test, y_train, y_test, predictions, {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def save_outputs(metrics_dict, actual_values, predicted_values, student_info):
    """Save evaluation results and prediction outputs to the output folder."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(
        [{
            "model": "LinearRegression",
            "MAE": metrics_dict["MAE"],
            "RMSE": metrics_dict["RMSE"],
            "R2": metrics_dict["R2"],
        }]
    )
    metrics_df.to_csv(METRICS_PATH, index=False)

    prediction_df = pd.DataFrame({
        "network": student_info["network"].tolist(),
        "student_id": student_info["student_id"].tolist(),
        "label": student_info["label"].tolist(),
        "actual_total_weight": actual_values.tolist(),
        "predicted_total_weight": predicted_values.tolist(),
    })
    prediction_df.to_csv(PREDICTIONS_PATH, index=False)

    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved predictions: {PREDICTIONS_PATH}")


def main():
    """Run the full Linear Regression workflow."""
    print("Loading student activity data...")
    student_df = load_student_activity_data()

    print("Preparing features and target...")
    X, y, prepared_df = prepare_features_and_target(student_df)

    if X.empty or y.empty:
        print(
            "Warning: The dataset does not contain enough valid rows for a meaningful "
            "regression model. Please check the output/student_activity.csv file."
        )
        return

    print("Training Linear Regression model...")
    model, _, X_test, _, y_test, predictions, metrics = train_and_evaluate_model(X, y)
    save_model_visuals(model, X_test, y_test, predictions)

    test_results = prepared_df.loc[X_test.index, ["network", "student_id", "label", "total_weight"]].copy().reset_index(drop=True)
    test_results["actual_total_weight"] = y_test.reset_index(drop=True).values
    test_results["predicted_total_weight"] = pd.Series(predictions).reset_index(drop=True).values

    save_outputs(metrics, test_results["actual_total_weight"], test_results["predicted_total_weight"], test_results)

    print("\nLinear Regression Evaluation")
    print("-" * 40)
    print(f"MAE:  {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"R2:   {metrics['R2']:.4f}")

    print("\nSample predictions:")
    print(test_results[["network", "student_id", "actual_total_weight", "predicted_total_weight"]].head().to_string(index=False))

    print("\nModel trained successfully.")


if __name__ == "__main__":
    main()
