from pydantic import BaseModel
from typing import List, Optional


class CreditRiskRequest(BaseModel):
    Amount: float
    Value: float
    trans_hour: int
    trans_day: int
    trans_month: int
    trans_year: int
    # Categorical fields (convert to strings)
    AccountId: str
    CustomerId: str
    CurrencyCode: str
    CountryCode: str
    ProviderId: str
    ProductCategory: str
    ChannelId: str
    PricingStrategy: str
    FraudResult: int


class CreditRiskResponse(BaseModel):
    risk_probability: float
    model_version: Optional[str] = None
