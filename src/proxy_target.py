# src/proxy_target.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime


def compute_rfm(df, customer_id="CustomerId", date_col="TransactionStartTime", amount_col="Amount"):
    """
    Computes Recency, Frequency, Monetary (RFM) metrics for each customer.
    """

    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")

    # Snapshot date = max transaction date + 1 day
    snapshot_date = data[date_col].max() + pd.Timedelta(days=1)

    rfm = data.groupby(customer_id).agg({
        date_col: lambda x: (snapshot_date - x.max()).days,   # Recency
        customer_id: "count",                                 # Frequency
        amount_col: "sum"                                     # Monetary
    })

    rfm.columns = ["recency", "frequency", "monetary"]
    rfm = rfm.reset_index()

    return rfm


def cluster_rfm(rfm_df, n_clusters=3, random_state=42):
    """
    Performs KMeans clustering on RFM data to create customer segments.
    """

    features = ["recency", "frequency", "monetary"]

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm_df["cluster"] = kmeans.fit_predict(rfm_scaled)

    return rfm_df, kmeans, scaler


def identify_high_risk_cluster(rfm_df):
    """
    Finds the cluster with lowest engagement:
    High recency (inactive), low frequency, low monetary = highest risk.
    """

    cluster_stats = rfm_df.groupby("cluster").agg({
        "recency": "mean",
        "frequency": "mean",
        "monetary": "mean"
    })

    # High recency + low frequency + low monetary → risky
    cluster_stats["risk_score"] = (
        cluster_stats["recency"] -
        cluster_stats["frequency"] -
        cluster_stats["monetary"]
    )

    high_risk_cluster = cluster_stats["risk_score"].idxmax()
    return high_risk_cluster, cluster_stats


def assign_high_risk_label(rfm_df, high_risk_cluster):
    """
    Creates binary target variable: 1 = high risk, 0 = low risk.
    """

    rfm_df["is_high_risk"] = (rfm_df["cluster"] == high_risk_cluster).astype(int)
    return rfm_df


def merge_target_with_features(processed_features_df, rfm_df, customer_id="CustomerId"):
    """
    Merges the high-risk target label back into the main processed dataset.
    """

    merged = processed_features_df.merge(
        rfm_df[[customer_id, "is_high_risk"]],
        on=customer_id,
        how="left"
    )

    merged["is_high_risk"] = merged["is_high_risk"].fillna(0).astype(int)
    return merged


def generate_proxy_target(df_raw, df_processed):
    """
    Full pipeline:
    1. Compute RFM
    2. Cluster
    3. Identify high-risk cluster
    4. Assign target
    5. Merge back into processed features
    """

    rfm = compute_rfm(df_raw)
    rfm_clustered, kmeans, scaler = cluster_rfm(rfm)
    high_risk_cluster, stats = identify_high_risk_cluster(rfm_clustered)
    rfm_labeled = assign_high_risk_label(rfm_clustered, high_risk_cluster)

    merged = merge_target_with_features(df_processed, rfm_labeled)

    return merged, rfm_labeled, stats, kmeans, scaler

