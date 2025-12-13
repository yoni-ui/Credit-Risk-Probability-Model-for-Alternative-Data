import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def preprocess_raw_data(df: pd.DataFrame):
    """
    Full preprocessing pipeline:
    - Handle datetimes
    - Extract hour/day/month/year
    - Remove non-informative columns
    - Apply scaling + OHE
    """

    df = df.copy()

    # Ensure timestamp is parsed
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    # Time features
    df["trans_hour"] = df["TransactionStartTime"].dt.hour
    df["trans_day"] = df["TransactionStartTime"].dt.day
    df["trans_month"] = df["TransactionStartTime"].dt.month
    df["trans_year"] = df["TransactionStartTime"].dt.year

    # Drop unused columns (ID + timestamp)
    drop_cols = [
        "TransactionId", "BatchId", "SubscriptionId",
        "TransactionStartTime", "ProductId"
    ]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Numerical & categorical
    numeric_features = [
        "Amount",
        "Value",
        "trans_hour", "trans_day", "trans_month", "trans_year"
    ]

    categorical_features = [
        "AccountId", "CustomerId", "CurrencyCode", "CountryCode",
        "ProviderId", "ProductCategory", "ChannelId",
        "PricingStrategy", "FraudResult"
    ]

    # Column Transformer
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Fit + Transform
    transformed = preprocessor.fit_transform(df)

    # Get OHE output names dynamically
    ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    ohe_cols = list(ohe.get_feature_names_out(categorical_features))

    # Full column list
    processed_col_names = numeric_features + ohe_cols

    # Build final DataFrame
    processed_df = pd.DataFrame(transformed.toarray(), columns=processed_col_names)

    # Include original index to merge later
    processed_df["CustomerId"] = df["CustomerId"].values

    return processed_df
