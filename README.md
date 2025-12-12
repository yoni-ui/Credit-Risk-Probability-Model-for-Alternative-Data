Credit Scoring Business Understanding
1. How Basel II influences our need for an interpretable and well-documented model

The Basel II Accord requires financial institutions to use transparent, auditable, and well-justified methods for estimating credit risk.
Banks must demonstrate:

How a model produces its predictions

Why the model is reliable

How risk is measured, monitored, and validated

How decisions can be explained to auditors and regulators

Because of this, highly interpretable models such as Logistic Regression + Weight of Evidence (WoE) are traditionally preferred. They allow:

Clear explanation of how each feature influences the probability of default

Reproducibility and traceability

Easy documentation for risk committees

In this challenge, Basel II influences our approach by requiring:

✔️ A well-defined proxy target
✔️ Documented feature engineering
✔️ Transparent modeling steps
✔️ Version-controlled experiments (MLflow)
✔️ Explainable predictions for approval decisions

2. Why a proxy “default” variable is needed, and its business risks

The dataset does not contain a true default label.
Therefore, we must construct a proxy variable that approximates credit risk.

We use RFM (Recency, Frequency, Monetary) customer behavior to identify disengaged customers who resemble high-risk borrowers.

Why the proxy is necessary:

Required for supervised learning

Enables model training without explicit loan data

Allows segmentation of customers into likely good vs high-risk groups

Provides early insights for a Buy-Now-Pay-Later (BNPL) product launch

Business risks of using a proxy label:

Risk	Explanation
Misclassification	Customers labeled “high-risk” may actually be good, and vice versa.
Bias introduction	Behavioral patterns may not fully represent repayment behavior.
Regulatory scrutiny	Proxy-based decisions must be validated before real lending.
Reputational damage	Incorrect denial of credit harms customer trust.

The proxy is useful for experimentation, but must be validated or replaced with real repayment data before production deployment.

3. Trade-off: Interpretable vs. High-Performance Models
Simple (Logistic Regression + WoE)	Complex (Random Forest, XGBoost, Gradient Boosting)
High interpretability	Lower interpretability
Easy to justify to regulators	Hard to explain feature contributions
Stable & predictable	Potential overfitting
Follows traditional banking practices	Higher accuracy, especially with nonlinear interactions
Easy to monitor & maintain	Requires careful tuning and monitoring

In a regulated financial environment:

Interpretability and documentation are equally important as accuracy.

Complex models may be allowed only with strong governance and SHAP-based explainability.

Basel II alignment favors interpretable models for initial deployment.