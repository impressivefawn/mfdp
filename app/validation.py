from pydantic import BaseModel
from datetime import datetime
from models import Gender
from typing import Optional


class LoanApplicationCreate(BaseModel):
    client_email: str
    loan_amount: float = 10000.00
    application_date: Optional[datetime] = None
    repayment_term_periods: int = 4


class MacroeconomicDataResponse(BaseModel):
    reporting_date: datetime
    money_supply_m2: float
    unemployment_rate_male: float
    unemployment_rate_female: float
    consumer_confidence: float
    export_prices: float
    usdtwd: float
    twblr: float

    class Config:
        orm_mode = True


class EarningsResponse(BaseModel):
    age_group: str
    gender: Gender
    median_earning: float

    class Config:
        orm_mode = True


class ClientResponse(BaseModel):
    id: int
    email: str
    phone: str
    date_of_birth: datetime
    gender: Gender
    full_name: str
    created_at: datetime
    valid_from: datetime
    valid_to: Optional[datetime]
    is_current: bool
