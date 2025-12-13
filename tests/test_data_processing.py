from src.data_processing import preprocess_raw_data
import pandas as pd


def test_preprocess_output_columns():
    sample = {
        "TransactionStartTime": ["2021-01-01 12:00:00"],
        "Amount": [100],
        "Value": [100],
        "AccountId": ["A1"],
        "CustomerId": ["C1"],
        "CurrencyCode": ["UGX"],
        "CountryCode": ["256"],
        "ProviderId": ["P1"],
        "ProductCategory": ["Grocery"],
        "ChannelId": ["Web"],
        "PricingStrategy": ["Standard"],
        "FraudResult": [0],
    }

    df = pd.DataFrame(sample)
    processed = preprocess_raw_data(df)
    assert processed.shape[0] == 1
    assert "CustomerId" in processed.columns
