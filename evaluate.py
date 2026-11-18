"""Evaluate the coursework classifier and a browsing-feature ablation.

This script is a portfolio reproducibility addition to the supplied MP8 model.
It does not change the classifier or tune parameters on the public test set.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from threadpoolctl import threadpool_limits

from main import UserPredictor


def load_split(data_dir, split):
    users = pd.read_csv(data_dir / f"{split}_users.csv")
    logs = pd.read_csv(data_dir / f"{split}_logs.csv")
    labels = pd.read_csv(data_dir / f"{split}_y.csv")
    if users["user_id"].duplicated().any() or labels["user_id"].duplicated().any():
        raise ValueError(f"Duplicate user IDs in the {split} split.")
    if set(users["user_id"]) != set(labels["user_id"]):
        raise ValueError(f"User IDs and label IDs differ in the {split} split.")
    if not logs["user_id"].isin(users["user_id"]).all():
        raise ValueError(f"Unknown user ID in {split} browsing logs.")
    labels = labels.set_index("user_id").loc[users["user_id"]].reset_index()
    if not labels["y"].isin([0, 1]).all():
        raise ValueError("The target must contain only binary labels.")
    return users, logs, labels


def summarize(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path("evaluation.json"))
    args = parser.parse_args()
    train_users, train_logs, train_y = load_split(args.data_dir, "train")
    test_users, test_logs, test_y = load_split(args.data_dir, "test1")
    if set(train_users["user_id"]) & set(test_users["user_id"]):
        raise ValueError("Train and test user IDs overlap.")

    majority = int(train_y["y"].mode().iloc[0])
    metrics = {
        "majority_class_baseline": summarize(
            test_y["y"], np.full(len(test_y), majority, dtype=int)
        )
    }
    with threadpool_limits(limits=2):
        for model_name, use_logs in [
            ("user_profiles_only", False),
            ("user_profiles_and_browsing", True),
        ]:
            model = UserPredictor()
            fit_logs = train_logs if use_logs else train_logs.iloc[:0]
            predict_logs = test_logs if use_logs else test_logs.iloc[:0]
            model.fit(train_users, fit_logs, train_y)
            predictions = model.predict(test_users, predict_logs)
            metrics[model_name] = summarize(test_y["y"], predictions)
            print(f"{model_name}: {json.dumps(metrics[model_name])}")

    report = {
        "evaluation_set": "course-provided public test1 split",
        "train_users": len(train_users),
        "train_log_rows": len(train_logs),
        "test_users": len(test_users),
        "test_log_rows": len(test_logs),
        "train_positive_rate": float(train_y["y"].mean()),
        "test_positive_rate": float(test_y["y"].mean()),
        "metrics": metrics,
        "confusion_matrix_order": "rows = actual [0, 1]; columns = predicted [0, 1]",
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "limitations": [
            "The dataset was generated for a course exercise.",
            "Public test1 performance is not evidence of production marketing impact.",
            "This evaluation adds baselines and a feature ablation; it does not perform cross-validation.",
            "The supplied classifier retains the names field as a categorical input; its relevance and stability need further assessment.",
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Evaluation complete; JSON report saved.")


if __name__ == "__main__":
    main()
