# src/data_processing.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from datetime import datetime


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts date features from a datetime column.
    """
    def __init__(self, datetime_col="TransactionStartTime"):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df[self.datetime_col] = pd.to_datetime(df[self.datetime_col], errors="coerce")

        df["transaction_hour"] = df[self.datetime_col].dt.hour
        df["transaction_day"] = df[self.datetime_col].dt.day
        df["transaction_month"] = df[self.datetime_col].dt.month
        df["transaction_year"] = df[self.datetime_col].dt.year

        return df.drop(columns=[self.datetime_col])


class CustomerAggregator(BaseEstimator, TransformerMixin):
    """
    Aggregates transaction-level data into customer-level behavioral features.
    """
    def __init__(self, id_col="CustomerId", amount_col="Amount", value_col="Value"):
        self.id_col = id_col
        self.amount_col = amount_col
        self.value_col = value_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        agg_df = df.groupby(self.id_col).agg({
            self.amount_col: ["sum", "mean", "std", "count"],
            self.value_col: ["sum", "mean"]
        })

        agg_df.columns = [
            "total_amount",
            "avg_amount",
            "std_amount",
            "transaction_count",
            "total_value",
            "avg_value"
        ]

        agg_df = agg_df.reset_index()
        return agg_df


class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    Applies Label Encoding to selected categorical columns.
    Suitable for high-cardinality columns.
    """
    def __init__(self, columns):
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        for col in self.columns:
            le = LabelEncoder()
            X[col] = X[col].astype(str)
            le.fit(X[col])
            self.encoders[col] = le
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].astype(str)
            X[col] = self.encoders[col].transform(X[col])
        return X


def build_feature_pipeline():

    date_pipeline = Pipeline([
        ("extract_dates", DateFeatureExtractor("TransactionStartTime"))
    ])

    # Low-cardinality categorical columns for OneHotEncoding
    categorical_cols_ohe = ["CurrencyCode", "ChannelId", "ProductCategory", "PricingStrategy"]

    # High-cardinality columns for Label Encoding
    categorical_cols_label = ["ProviderId", "ProductId"]

    numeric_cols = ["Amount", "Value"]

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])

    categorical_ohe_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # LabelEncoderTransformer handles high-cardinality encodings
    label_encoding = LabelEncoderTransformer(columns=categorical_cols_label)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat_ohe", categorical_ohe_pipeline, categorical_cols_ohe)
        ],
        remainder="passthrough"
    )

    full_pipeline = Pipeline([
        ("date_features", date_pipeline),
        ("label_encoding", label_encoding),
        ("preprocessing", preprocessor)
    ])

    return full_pipeline


def preprocess_raw_data(df):
    """
    Performs full preprocessing on raw transaction data.
    Does NOT aggregate yet; used for model input preparation.
    """
    pipeline = build_feature_pipeline()
    transformed = pipeline.fit_transform(df)

    processed_col_names = (
        ["Amount_scaled", "Value_scaled"]
        + list(pipeline.named_steps["preprocessing"].transformers_[1][1]
        .named_steps["ohe"].get_feature_names_out(["CurrencyCode", "ChannelId", "ProductCategory", "PricingStrategy"]))
        + ["ProviderId_encoded", "ProductId_encoded"]
        + ["transaction_hour", "transaction_day", "transaction_month", "transaction_year"]
        + ["CustomerId", "BatchId", "SubscriptionId", "CountryCode", "FraudResult"]
    )

    processed_df = pd.DataFrame(transformed, columns=processed_col_names)
    return processed_df


def create_customer_aggregate_features(df):
    """
    Aggregates per-customer features for modeling.
    """
    transformer = CustomerAggregator()
    return transformer.transform(df)

