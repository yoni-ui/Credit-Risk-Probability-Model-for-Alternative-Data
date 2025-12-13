# --- PATH FIX START ---
import os
import sys

# Calculate the path to the project root (one directory up from 'src')
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')

# Add the project root to the system path
if project_root not in sys.path:
    sys.path.append(project_root)
# --- PATH FIX END ---

import pandas as pd # Line 3 now
import mlflow
# ... rest of your imports
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from src.data_processing import preprocess_raw_data
from src.proxy_target import generate_proxy_target


def load_data(raw_path="data/raw/data.csv"):
    """Load raw transaction data."""
    df = pd.read_csv(raw_path)
    return df


def prepare_training_data(df_raw):
    """Preprocess and generate proxy target."""
    processed = preprocess_raw_data(df_raw)
    final_df, rfm_labeled, stats, kmeans, scaler = generate_proxy_target(df_raw, processed)
    return final_df


def split_data(df, target_col="is_high_risk", test_size=0.2, random_state=42):
    """Split processed dataset into train and test sets."""

    columns_to_drop = [
        target_col,
        "CustomerId",   # prevent leakage
    ]

    # X = all features except target + ID
    X = df.drop(columns=columns_to_drop, errors="ignore")

    # y = target variable
    y = df[target_col]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)



def evaluate_model(model, X_test, y_test):
    """Compute evaluation metrics."""
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs)
    }


def run_logistic_regression(X_train, y_train):
    """Train LR with GridSearch."""
    model = LogisticRegression(max_iter=2000)

    param_grid = {
        "C": [0.1, 1.0, 10],
        "penalty": ["l2"],
        "solver": ["liblinear", "lbfgs"]
    }

    grid = GridSearchCV(model, param_grid, cv=3, scoring="roc_auc")
    grid.fit(X_train, y_train)

    return grid.best_estimator_


def run_random_forest(X_train, y_train):
    """Train RF with basic hyperparameter tuning."""
    model = RandomForestClassifier()

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
    }

    grid = GridSearchCV(model, param_grid, cv=3, scoring="roc_auc")
    grid.fit(X_train, y_train)

    return grid.best_estimator_


def log_model_to_mlflow(model_name, model, metrics, X_train):
    """Log model, metrics, and artifacts to MLflow."""

    with mlflow.start_run(run_name=model_name):
        # Log metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # Log parameters
        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Register model to model registry
        mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
            name=f"credit_risk_{model_name}"
        )

        mlflow.end_run()


def main():
    print("📂 Loading raw data...")
    df_raw = load_data()

    # --- ADD THIS LINE FOR INSPECTION ---
    print("Column names in raw data:", df_raw.columns.tolist()) 
    # ------------------------------------

    print("⚙️ Preprocessing data and generating proxy target...")
    df_final = prepare_training_data(df_raw)

    print("✂ Splitting train and test...")
    X_train, X_test, y_train, y_test = split_data(df_final)

    mlflow.set_experiment("credit_risk_modeling")

    # -------------------------
    # Logistic Regression
    # -------------------------
    print("🔍 Training Logistic Regression...")
    lr_model = run_logistic_regression(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test)

    print("📊 LR Metrics:", lr_metrics)
    log_model_to_mlflow("logistic_regression", lr_model, lr_metrics, X_train)

    # -------------------------
    # Random Forest
    # -------------------------
    print("🌲 Training Random Forest...")
    rf_model = run_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test)

    print("📊 RF Metrics:", rf_metrics)
    log_model_to_mlflow("random_forest", rf_model, rf_metrics, X_train)

    print("✅ Task 5 complete — models trained, tracked, and registered in MLflow!")


if __name__ == "__main__":
    main()
