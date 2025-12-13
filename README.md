🚀 Credit Risk Probability Model for Alternative Data (B8W4 Final Report)

10 Academy – Week 4 Challenge
Author: Yonas Yishak
Date: December 16, 2025

1. Introduction

Bati Bank is partnering with a fast-growing eCommerce company to roll out a Buy-Now-Pay-Later (BNPL) service. This project delivers a credit scoring model capable of evaluating a customer’s likelihood of default using only alternative transactional data.

Challenge Context

No loan repayment history available.

No default labels exist.

Only alternative transaction data provided.

Project Objectives (E2E Pipeline)

Define a proxy variable for credit risk.

Identify predictive behavioral features.

Build a risk probability model.

Produce a credit score (Future Work).

Recommend loan amount and duration (Future Work).

Deploy the model via FastAPI.

Automate testing with CI/CD.

2. Task 1 — Credit Scoring Business Understanding

2.1 Basel II and Interpretability

Regulatory compliance requires models to be transparent and auditable.

Key Insight

Requirement

Interpretability is mandatory

Necessary for regulatory compliance (Basel II).

Logistic Regression + WoE

Provides monotonic transformations and predictable feature influence.

Complex Models (XGBoost)

Require extensive explanation layers for regulators.

2.2 Proxy Target Variable Necessity

Since no default labels existed, a behavioral proxy was engineered using RFM clustering.

Component

Steps

RFM-based Clustering

1. Compute RFM metrics. 2. Apply KMeans clustering (k=3). 3. Label the least active cluster as high-risk (1). 4. Label others as low-risk (0).

Potential Risk

Description

Misclassification

Behavioral disengagement ≠ actual default.

Bias

Proxy encodes customer engagement, not creditworthiness.

Regulatory Concerns

Proxy cannot replace actual risk data in final production.

2.3 Simple vs. Complex Models Trade-offs

Aspect

Logistic Regression + WoE (Champion)

Complex Models (XGBoost, RF)

Explainability

Easy

Harder

Predictive Power

Moderate

High

Basel II Compliance

Fully compliant

Needs monitoring

Stability

Predictable

Risk of overfitting

Key Insight: Financial institutions benefit from a hybrid approach: interpretable models for decisioning and complex models for internal benchmarking.

3. Task 2 — Exploratory Data Analysis (EDA)

EDA was conducted in notebooks/eda.ipynb.

3.1 Dataset Overview

Rows: ~330,000

Features: Amount, Value, TransactionStartTime, ProviderId (High-Cardinality), ProductId (High-Cardinality).

3.7 Top Insights from EDA

Transaction Amounts Are Extremely Skewed: Requires robust scaling.

Customer Behavior Varies Significantly: Validates the use of RFM segmentation.

Missing Categorical Values: Handled via mode imputation or a separate WoE placeholder category.

High-Cardinality Features: Avoided one-hot encoding; used WoE encoding instead.

Weak Correlations Among Raw Features: Engineered features are critical for predictive modeling.

4. Task 3 & 4 — Feature Engineering and Proxy Target

Task 3: Feature Engineering

Aggregate Features: Sum, mean, count, and standard deviation per customer.

Temporal Features: Transaction hour, day, month, year extraction.

Encoding: Label and WoE encoding for categorical variables.

Scaling: RobustScaler applied to skewed monetary features.

Key Insight: Engineered features provide significant predictive power over raw transaction data.

Task 4: Proxy Target Engineering

The RFM metrics were computed, clustered using K-Means (k=3), and the resultant high-risk flag was merged into the training dataset.

Key Insight: The proxy target enables supervised learning despite the absence of historical default labels.

5. Task 5 — Model Training & Tracking

Experimentation and Selection

Data Split: 80% training, 20% test (fixed random_state).

Models: Logistic Regression (Champion) and Gradient Boosting (LightGBM/XGBoost).

Tracking: All runs, parameters, and metrics were tracked with MLflow.

Result: The Logistic Regression model was selected and registered in the MLflow registry. It achieved a high ROC-AUC while maintaining the required interpretability for financial regulation.

6. Task 6 — Deployment & CI/CD

FastAPI Model Serving

The final model is deployed as a containerized REST API, ensuring reproducibility and scalability.

Service: FastAPI with the /predict endpoint.

Validation: Pydantic models validate incoming requests.

Containerization: Docker containerization and docker-compose setup.

Example API Request Body (POST /predict)

This JSON must contain all features required by the Pydantic schema (including all encoded and engineered values).

{
  "Amount": 1500.0,
  "Value": 1500.0,
  "PricingStrategy": 1,
  "TransactionHour": 14,
  
  "Recency": 10,
  "Frequency": 5,
  "Monetary": 8000.0,
  "is_high_value": 1,
  
  "AccountId": "A-7343",         
  "CustomerId": "C-123456",
  "CurrencyCode": "USD",
  "CountryCode": 256,
  "ProviderId": "P-100",
  "ProductCategory": "Utility",
  "ChannelId": "Mobile",
  "FraudResult": 0
  
  // NOTE: All other 30+ features must be sent in the request body.
}


Key Insight: Containerized API ensures reproducibility, automated deployment, and easy integration with production systems.

CI/CD Pipeline

A GitHub Actions workflow manages Continuous Integration, ensuring code quality before deployment.

Checks: Linting (flake8/black) and unit test execution (pytest).

7. Discussion and Conclusion

The end-to-end ML pipeline is complete, production-ready, and aligned with both business objectives and Basel II regulatory requirements.

Proxy target allows supervised modeling in the absence of default labels.

RFM features and aggregate metrics are critical for predictive performance.

Interpretable models comply with Basel II while providing transparency.

Containerized FastAPI deployment ensures reproducibility and scalability.

The system provides a robust foundation for credit risk prediction using alternative data and is ready for deployment with continuous monitoring.

8. References

Basel II Accord: https://www.bis.org/publ/bcbs128.pdf

Alternative Credit Scoring: HKMA

WoE & IV Packages: xverse, woe

Dataset: Xente Challenge | Kaggle
