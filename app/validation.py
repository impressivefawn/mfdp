from pydantic import BaseModel, field_validator
from datetime import datetime
from models import Gender
from typing import Optional
import re


class LoanApplicationCreate(BaseModel):
    client_email: str
    loan_amount: float = 10000.00
    application_date: Optional[datetime] = None
    repayment_term_periods: int = 4

    @field_validator('client_email', mode='before')
    @staticmethod
    def validate_email(email: str) -> str:
        """Validate email format."""
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_regex, email):
            raise ValueError(f"Invalid email format: {email}")
        return email

    @field_validator('repayment_term_periods', mode='before')
    @staticmethod
    def validate_repayment_term_periods(periods: int) -> int:
        allowed_periods = [4, 6, 8]
        if periods not in allowed_periods:
            raise ValueError(f"Repayment term periods must be one of {allowed_periods}. Given: {periods}")
        return periods

    @field_validator('loan_amount', mode='before')
    @staticmethod
    def validate_loan_amount(amount: float) -> float:
        # Check if amount is within the specified range
        if not (1000 <= amount <= 15000):
            raise ValueError(f"Loan amount must be between 1000 and 15000. Given: {amount}")

        # Check if amount has more than two decimal places
        if isinstance(amount, float) and round(amount, 2) != amount:
            raise ValueError(f"Loan amount must have at most two decimal places. Given: {amount}")

        return amount


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
