"""Marketing email click prediction.

COMP SCI 320 MP8 coursework implementation, Pengtian Sun.
Uses user profiles and aggregated browsing histories.
"""

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


class UserPredictor:
    def __init__(self):
        self.model = None
        self.xcols = None

    def add_logs_as_features(self, users, logs):
        users = users.copy()
        logs = logs.copy()

        if len(logs) == 0:
            users["total_seconds"] = 0
            users["num_visits"] = 0
            users["avg_seconds"] = 0
            users["max_seconds"] = 0
            users["visited_laptop"] = 0
            return users

        summary = logs.groupby("user_id").agg(
            total_seconds=("seconds", "sum"),
            num_visits=("seconds", "count"),
            avg_seconds=("seconds", "mean"),
            max_seconds=("seconds", "max"),
        )

        laptop_visits = (
            logs.assign(
                visited_laptop=logs["url"].astype(str)
                .str.contains("laptop", case=False, na=False)
                .astype(int)
            )
            .groupby("user_id")["visited_laptop"]
            .max()
        )

        summary = summary.join(laptop_visits, how="left")
        summary["visited_laptop"] = summary["visited_laptop"].fillna(0)

        users = users.merge(summary, how="left", left_on="user_id", right_index=True)

        for col in ["total_seconds", "num_visits", "avg_seconds", "max_seconds", "visited_laptop"]:
            users[col] = users[col].fillna(0)

        return users

    def fit(self, train_users, train_logs, train_y):
        train_users = self.add_logs_as_features(train_users, train_logs)

        X = train_users.drop(columns=["user_id"], errors="ignore")
        y = train_y["y"]

        self.xcols = X.columns.tolist()

        numeric_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()

        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ])

        self.model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ])

        self.model.fit(X, y)

    def predict(self, test_users, test_logs):
        test_users = self.add_logs_as_features(test_users, test_logs)

        X = test_users.drop(columns=["user_id"], errors="ignore")

        for col in self.xcols:
            if col not in X.columns:
                X[col] = 0

        X = X[self.xcols]

        return self.model.predict(X)
