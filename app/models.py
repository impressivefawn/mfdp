from enum import Enum
import re
import bcrypt
from datetime import datetime
from sqlmodel import SQLModel, Field, Relationship
from typing import List, Optional


class ApplicationStatus(Enum):
    APPROVED = 'APPROVED'
    DENIED = 'DENIED'


class Gender(Enum):
    MALE = 'M'
    FEMALE = 'F'


class Client(SQLModel, table=True):
    """Represents a client.

    Attributes:
        id (Optional[int]): The unique identifier for the client.
        email (str): The email address of the client, must be unique.
        phone (str): The phone number of the client.
        date_of_birth (datetime): The birth date of the client.
        gender (Gender): The gender of the client, represented as an enum.
        full_name (str): The full name of the client.
        password_hash (str): The hashed password for client authentication.
        created_at (datetime): The timestamp when the client was created.
        valid_from (datetime): The start date for the client's validity.
        valid_to (Optional[datetime]): The end date for the client's validity.
        is_current (bool): Indicates whether the client's information is current.
        loan_applications (List[LoanApplication]): A list of loan applications associated with the client.

    Methods:
        validate_email(email: str) -> None: Validates the format of the email address.
        validate_phone(phone: str) -> None: Validates the format of the phone number.
        hash_password(password: str) -> str: Hash the given password using bcrypt and return the hashed value
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(max_length=255, nullable=False, unique=True)
    phone: str = Field(default=None, max_length=15)
    date_of_birth: datetime = Field(nullable=False)
    gender: Gender = Field(nullable=False)
    full_name: str = Field(max_length=255, nullable=False)
    password_hash: str = Field(max_length=255, nullable=False)
    created_at: datetime = Field(default_factory=datetime.now)
    valid_from: datetime = Field(default_factory=datetime.now)
    valid_to: Optional[datetime] = Field(default=None)
    is_current: bool = Field(default=True)

    loan_applications: List["LoanApplication"] = Relationship(back_populates="client")

    def __init__(self, password: str, **data):
        super().__init__(**data)
        self.validate_email(self.email)
        self.validate_phone(self.phone)
        self.password_hash = self.hash_password(password)

    @staticmethod
    def hash_password(password: str) -> str:
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed_password.decode('utf-8')

    @staticmethod
    def validate_email(email: str) -> None:
        """Validate email format."""
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_regex, email):
            raise ValueError(f"Invalid email format: {email}")

    @staticmethod
    def validate_phone(phone: str) -> None:
        """Validate phone format."""
        phone_regex = r'^\+7 \d{3}-\d{3}-\d{4}$'
        if not re.match(phone_regex, phone):
            raise ValueError(f"Invalid phone format: {phone}")


class LoanApplication(SQLModel, table=True):
    """Represents a loan application submitted by a client.

    Attributes:
        id (Optional[int]): Unique identifier for each loan application.
        client_id (int): Foreign key linking to the associated client's ID.
        loan_amount (float): The amount of the loan requested by the client.
        application_date (datetime): The date when the loan application was submitted.
        application_status (ApplicationStatus): The current status of the application (e.g., approved, denied).
        repayment_term_periods (int): The number of payment periods for the loan repayment.
        client (Client): The client associated with this loan application.
        repayment_schedules (List[RepaymentSchedule]): A list of repayment schedules associated with this loan application.

    Methods:
        validate_loan_amount(amount: float) -> None: Validate the loan amount to ensure it is within the specified range and has two decimal places.
        validate_repayment_term_periods(periods: int) -> None: "Validate repayment term periods to ensure it is one of the allowed options.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    client_id: int = Field(foreign_key="client.id")
    loan_amount: float = Field(nullable=False)
    application_date: datetime = Field(default_factory=datetime.now)
    application_status: ApplicationStatus = Field(nullable=False)
    repayment_term_periods: int = Field(nullable=False)

    client: Client = Relationship(back_populates="loan_applications")
    repayment_schedules: List["RepaymentSchedule"] = Relationship(back_populates="loan_application")

    def __init__(self, **data):
        super().__init__(**data)
        self.validate_loan_amount(self.loan_amount)
        self.validate_repayment_term_periods(self.repayment_term_periods)

    @staticmethod
    def validate_loan_amount(amount: float) -> None:
        if not (1000 <= amount <= 15000):
            raise ValueError(f"Loan amount must be between 1000 and 15000. Given: {amount}")

        # Check if amount has more than two decimal places
        if isinstance(amount, float) and round(amount, 2) != amount:
            raise ValueError(f"Loan amount must have at most two decimal places. Given: {amount}")

    @staticmethod
    def validate_repayment_term_periods(periods: int) -> None:
        allowed_periods = [4, 6, 8]
        if periods not in allowed_periods:
            raise ValueError(f"Repayment term periods must be one of {allowed_periods}. Given: {periods}")


class RepaymentSchedule(SQLModel, table=True):
    """Represents a repayment schedule for a loan application.

    Attributes:
        id (Optional[int]): Unique identifier for each repayment schedule.
        loan_application_id (int): Foreign key linking to the associated loan application's ID.
        period_number (int): The number of the payment period (e.g., 1, 2, 3).
        due_date (datetime): The date when the payment is due.
        amount (float): The amount to be paid in this period.
        status (str): The current status of the repayment (e.g., 'pending', 'completed').
        created_at (datetime): The date and time when this repayment schedule was created.
        loan_application (LoanApplication): The loan application associated with this repayment schedule.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    loan_application_id: int = Field(foreign_key="loanapplication.id")
    period_number: int = Field(nullable=False)
    due_date: datetime = Field(nullable=False)
    amount: float = Field(nullable=False)
    status: str = Field(default="pending", max_length=20)
    created_at: datetime = Field(default_factory=datetime.now)

    loan_application: LoanApplication = Relationship(back_populates="repayment_schedules")


class Earnings(SQLModel, table=True):
    """Represents earnings data segmented by age group and gender."""

    id: Optional[int] = Field(default=None, primary_key=True)
    age_group: str = Field(max_length=50, nullable=False)
    gender: Gender = Field(nullable=False)
    median_earning: float = Field(nullable=False)
    valid_from: datetime = Field(default_factory=datetime.now)
    valid_to: Optional[datetime] = Field(default=None)
    is_current: bool = Field(default=True)


class MacroeconomicData(SQLModel, table=True):
    """Represents macroeconomic data for analysis."""

    id: Optional[int] = Field(default=None, primary_key=True)
    reporting_date: datetime = Field(nullable=False)
    money_supply_m2: float = Field(nullable=False)
    unemployment_rate_male: float = Field(nullable=False)
    unemployment_rate_female: float = Field(nullable=False)
    consumer_confidence: float = Field(nullable=False)
    export_prices: float = Field(nullable=False)
    usdtwd: float = Field(nullable=False)
    twblr: float = Field(nullable=False)
