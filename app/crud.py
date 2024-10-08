from sqlmodel import SQLModel, Session, select
from typing import List, Optional, Union
from models import Client, LoanApplication, ApplicationStatus, Earnings, MacroeconomicData, Gender
from db import engine
from datetime import datetime
from contextlib import contextmanager
import pandas as pd
from clearml import Task
from pandas.tseries.offsets import MonthEnd
from dateutil.relativedelta import relativedelta
import os
from catboost import CatBoostClassifier


@contextmanager
def get_session():
    session = Session(engine)
    try:
        yield session
    finally:
        session.close()


def create_tables():
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)


def create_clients(session: Session):
    """Create and add clients to the database using the provided session."""
    try:
        client_1 = Client(
            email='john.doe@example.com',
            phone='+7 999-123-4567',
            date_of_birth=datetime(1985, 6, 15),
            gender=Gender.MALE,
            full_name='John Doe',
            password='qwerty',
            valid_to=None
        )

        client_2 = Client(
            email='jane.smith@example.com',
            phone='+7 999-987-6543',
            date_of_birth=datetime(1990, 3, 22),
            gender=Gender.FEMALE,
            full_name='Jane Smith',
            password='qwerty',
            valid_to=None
        )

        client_3 = Client(
            email='emily.jones@example.com',
            phone='+7 999-555-1212',
            date_of_birth=datetime(1978, 11, 5),
            gender=Gender.FEMALE,
            full_name='Emily Jones',
            password='qwerty',
            valid_to=None
        )

        session.add(client_1)
        session.add(client_2)
        session.add(client_3)

        session.commit()
        print("Clients created successfully.")
    except Exception as e:
        print(f"Error creating clients: {e}")


def create_earnings(session: Session):
    """Create and add earnings data to the database using the provided session."""
    try:
        earnings_data = [
            ('24 years & below', Gender.MALE, 25358.9),
            ('24 years & below', Gender.FEMALE, 20372.6),
            ('25-29 years', Gender.MALE, 35764.9),
            ('25-29 years', Gender.FEMALE, 28732.3),
            ('30-39 years', Gender.MALE, 42148.3),
            ('30-39 years', Gender.FEMALE, 33860.6),
            ('40-49 years', Gender.MALE, 43809.8),
            ('40-49 years', Gender.FEMALE, 35195.3),
            ('50-64 years', Gender.MALE, 43285.1),
            ('50-64 years', Gender.FEMALE, 34773.8),
            ('65 years & above', Gender.MALE, 33578.7),
            ('65 years & above', Gender.FEMALE, 26976.1)
        ]

        for age_group, gender, median_earning in earnings_data:
            earning = Earnings(
                age_group=age_group,
                gender=gender,
                median_earning=median_earning,
                valid_from=datetime.now()
            )
            session.add(earning)

        session.commit()
        print("Earnings data created successfully.")
    except Exception as e:
        print(f"Error creating earnings data: {e}")


def create_macroeconomic_data(session: Session):
    """Create and add macroeconomic data to the database using the provided session."""
    try:
        macroeconomic_data = [
            (datetime(2024, 1, 31), 23036672, 4.48, 3.47, 75.85, 117.57, 31.939, 3.571),
            (datetime(2024, 2, 29), 23187107, 4.62, 3.80, 76.96, 116.74, 31.516, 3.584),
            (datetime(2024, 3, 31), 23235475, 4.51, 3.66, 75.13, 116.19, 31.164, 3.614),
            (datetime(2024, 4, 30), 23276293, 4.34, 3.62, 73.12, 116.60, 31.523, 3.657),
            (datetime(2024, 5, 31), 23196507, 4.28, 3.86, 73.06, 114.82, 31.303, 3.657),
            (datetime(2024, 6, 30), 23590734, 4.33, 4.08, 74.79, 114.83, 31.371, 3.671),
            (datetime(2024, 7, 31), 23647371, 4.42, 4.18, 73.15, 116.34, 31.921, 3.714),
            (datetime(2024, 8, 31), 23612780, 4.40, 4.30, 73.38, 116.84, 32.111, 3.734),
            (datetime(2024, 9, 30), 23692913, 4.09, 4.21, 71.42, 119.72, 32.924, 3.762),
            (datetime(2024, 10, 31), 23674589, 4.07, 4.07, 72.59, 120.83, 33.491, 3.810),
            (datetime(2024, 11, 30), 23909549, 4.07, 3.77, 74.11, 120.09, 33.580, 3.810),
            (datetime(2024, 12, 31), 24410078, 4.13, 3.49, 72.17, 118.69, 33.299, 3.845)
        ]

        for reporting_date, money_supply_m2, inflation_rate_male, inflation_rate_female,\
                consumer_confidence, exports_prices, twd_usd, twd_lr in macroeconomic_data:

            data = MacroeconomicData(
                reporting_date=reporting_date,
                money_supply_m2=money_supply_m2,
                unemployment_rate_male=inflation_rate_male,
                unemployment_rate_female=inflation_rate_female,
                consumer_confidence=consumer_confidence,
                export_prices=exports_prices,
                usdtwd=twd_usd,
                twblr=twd_lr
            )

            session.add(data)

        session.commit()
        print("Macroeconomic data created successfully.")
    except Exception as e:
        print(f"Error creating macroeconomic data: {e}")


def create_client(client: Client):
    """Create and add a single client to the database using a new session."""
    session = Session(engine)
    try:
        session.add(client)
        session.commit()
        print(f"Client created successfully: {client.email}")
        return client  # Optionally return the created client
    except Exception as e:
        session.rollback()
        print(f"Error creating client: {e}")
        raise  # Re-raise the exception for further handling
    finally:
        session.close()


def create_loan_application(loan_application: LoanApplication):
    """Create and add a single loan application to the database using a new session."""
    session = Session(engine)
    try:
        session.add(loan_application)
        session.commit()
        print(f"Loan application created successfully for client ID: {loan_application.client_id}")
        return loan_application
    except Exception as e:
        session.rollback()
        print(f"Error creating loan application: {e}")
        raise
    finally:
        session.close()


def get_client_by_id(client_id: int) -> Optional[Client]:
    """Retrieve a client by their ID using a new session."""
    session = Session(engine)
    try:
        statement = select(Client).where(Client.id == client_id)
        client = session.exec(statement).first()

        if client is None:
            raise ValueError(f"No client found with ID: {client_id}")

        return client
    except Exception as e:
        print(f"Error retrieving client by ID {client_id}: {e}")
        raise
    finally:
        session.close()


def get_client_by_email(email: str) -> Optional[Client]:
    """Retrieve a client by their email address using a new session."""
    session = Session(engine)
    try:
        statement = select(Client).where(Client.email == email)
        client = session.exec(statement).first()

        if client is None:
            raise ValueError(f"No client found with email: {email}")

        return client
    except Exception as e:
        print(f"Error retrieving client by email {email}: {e}")
        raise
    finally:
        session.close()


def get_loan_application(loan_application_id: int) -> Optional[LoanApplication]:
    """Retrieve a single loan application by its ID using a new session."""
    session = Session(engine)
    try:
        statement = select(LoanApplication).where(LoanApplication.id == loan_application_id)
        loan_application = session.exec(statement).first()

        if loan_application is None:
            raise ValueError(f"No loan application found with ID: {loan_application_id}")

        return loan_application
    except Exception as e:
        print(f"Error retrieving loan application with ID {loan_application_id}: {e}")
        raise
    finally:
        session.close()


def get_loan_applications_by_client(client_identifier: Union[int, str]) -> List[LoanApplication]:
    """Retrieve a list of loan applications for a client using either their client ID or email address."""
    session = Session(engine)
    try:
        if isinstance(client_identifier, str):
            # If it's an email, first get the client by email
            client = get_client_by_email(client_identifier)
            if not client:
                return []
            client_id = client.id
        elif isinstance(client_identifier, int):
            client_id = client_identifier
        else:
            raise ValueError("client_identifier must be either an int (client ID) or a str (email).")

        # Retrieve loan applications by client ID
        statement = select(LoanApplication).where(LoanApplication.client_id == client_id)
        return session.exec(statement).all()

    except Exception as e:
        print(f"Error retrieving loan applications for identifier {client_identifier}: {e}")
        raise
    finally:
        session.close()


def get_all_earnings() -> List[Earnings]:
    """Retrieve all current earnings records using a new session."""
    session = Session(engine)
    try:
        statement = select(Earnings).where(Earnings.is_current == True)
        earnings_records = session.exec(statement).all()
        return earnings_records
    except Exception as e:
        print(f"Error retrieving current earnings records: {e}")
        raise
    finally:
        session.close()


def get_all_clients() -> List[Client]:
    """Retrieve all current client records using a new session."""
    session = Session(engine)
    try:
        statement = select(Client).where(Client.is_current == True)
        client_records = session.exec(statement).all()
        return client_records
    except Exception as e:
        print(f"Error retrieving current client records: {e}")
        raise
    finally:
        session.close()


def get_all_macroeconomic_data() -> List[MacroeconomicData]:
    """Retrieve all macroeconomic data records using a new session."""
    session = Session(engine)
    try:
        statement = select(MacroeconomicData)
        macroeconomic_records = session.exec(statement).all()
        return macroeconomic_records
    except Exception as e:
        print(f"Error retrieving macroeconomic data records: {e}")
        raise
    finally:
        session.close()


def get_all_macroeconomic_data_sme3_lag1() -> List[MacroeconomicData]:
    """Calculate rolling average for 3 months with a 1-month lag."""
    session = Session(engine)
    try:
        # Retrieve all macroeconomic data records
        statement = select(MacroeconomicData)
        macroeconomic_records = session.exec(statement).all()

        # Convert records to DataFrame
        macro_df = pd.DataFrame([record.model_dump() for record in macroeconomic_records])
        macro_df.set_index('reporting_date', inplace=True)

        # Calculate rolling averages
        macro_sme3 = macro_df[[
            'money_supply_m2',
            'unemployment_rate_male',
            'unemployment_rate_female',
            'consumer_confidence',
            'export_prices',
            'usdtwd',
            'twblr'
        ]].rolling(window=3).mean()

        # Apply lag of 1 month
        macro_sme3_lag1 = macro_sme3.shift(1).reset_index()
        macro_sme3_lag1 = macro_sme3_lag1.iloc[3:]

        response_data = [
            MacroeconomicData(
                reporting_date=row['reporting_date'],
                money_supply_m2=row['money_supply_m2'],
                unemployment_rate_male=row['unemployment_rate_male'],
                unemployment_rate_female=row['unemployment_rate_female'],
                consumer_confidence=row['consumer_confidence'],
                export_prices=row['export_prices'],
                usdtwd=row['usdtwd'],
                twblr=row['twblr']
            )
            for _, row in macro_sme3_lag1.iterrows()
        ]

        return response_data

    except Exception as e:
        print(f"Error processing macroeconomic data: {e}")
        raise
    finally:
        session.close()


def get_age_group(age):
    if age <= 24:
        return '24 years & below'
    elif 25 <= age <= 29:
        return '25-29 years'
    elif 30 <= age <= 39:
        return '30-39 years'
    elif 40 <= age <= 49:
        return '40-49 years'
    elif 50 <= age <= 64:
        return '50-64 years'
    else:
        return '65 years & above'


best_model = None


def load_model():
    """Load the trained model from ClearML."""
    global best_model

    # Load ClearML configuration from environment variables with defaults
    clearml_api_host = os.getenv('CLEARML_API_HOST', 'https://api.clear.ml')
    clearml_web_host = os.getenv('CLEARML_WEB_HOST', 'https://api.clear.ml')
    clearml_files_host = os.getenv('CLEARML_FILES_HOST', 'https://files.clear.ml')
    clearml_access_key = os.getenv('CLEARML_ACCESS_KEY')
    clearml_secret_key = os.getenv('CLEARML_SECRET_KEY')

    # Check if access key and secret key are set
    if not clearml_access_key or not clearml_secret_key:
        raise ValueError("ClearML access key and secret key must be set.")

    # Set ClearML configuration
    os.environ['CLEARML_API_HOST'] = clearml_api_host
    os.environ['CLEARML_WEB_HOST'] = clearml_web_host
    os.environ['CLEARML_FILES_HOST'] = clearml_files_host
    os.environ['CLEARML_API_ACCESS_KEY'] = clearml_access_key
    os.environ['CLEARML_API_SECRET_KEY'] = clearml_secret_key

    # Load the trained model from ClearML
    task_model = Task.get_task(project_name='mfdp', task_name='Train and Save CatBoost Model')
    model_file_path = task_model.artifacts['Best CatBoost Model'].get_local_copy()

    # Load the model into the global variable
    best_model = CatBoostClassifier()
    best_model.load_model(model_file_path)


def data_preparation(client_email: str, loan_amount: float, application_date: datetime, repayment_term_periods: int) -> pd.DataFrame:
    """Prepare data for predicting application status based on client information and loan parameters."""

    # Retrieve client details using the provided email
    client = get_client_by_email(client_email)

    # Convert application_date to naive if it is aware
    if application_date.tzinfo is not None:
        application_date = application_date.replace(tzinfo=None)

    age = relativedelta(application_date, client.date_of_birth).years

    client_data = pd.DataFrame({
        "gender": [client.gender],
        "AGE": [age]
    })

    macro = get_all_macroeconomic_data_sme3_lag1()
    macro_sme3_lag1 = pd.DataFrame([record.model_dump() for record in macro])

    macro_sme3_lag1 = macro_sme3_lag1[[
        'reporting_date', 
        'money_supply_m2', 
        'unemployment_rate_male', 
        'unemployment_rate_female', 
        'consumer_confidence', 
        'export_prices', 
        'usdtwd', 
        'twblr'
    ]]

    macro_sme3_lag1.columns = [
        'DATE',
        'MONEY-SUPPLY-M2',
        'UNEMPLOYMENT-RATE-MALE',
        'UNEMPLOYMENT-RATE-FEMALE',
        'CONSUMER-CONFIDENCE',
        'EXPORT-PRICES',
        'USDTWD',
        'TWBLR'
    ]

    macro_sme3_lag1['DATE'] = pd.to_datetime(macro_sme3_lag1['DATE'])

    earnings_records = get_all_earnings()
    earnings_df = pd.DataFrame([record.model_dump() for record in earnings_records])

    age_group = get_age_group(age)
    client_data['age_group'] = age_group

    merged_data = client_data.merge(
        earnings_df[(earnings_df['age_group'] == age_group) & (earnings_df['gender'] == client.gender)],
        on=['age_group', 'gender'],
        how='left'
    )

    end_of_month_date = (application_date + MonthEnd(0)).date()

    # Ensure end_of_month is a datetime object
    merged_data['end_of_month'] = pd.to_datetime(end_of_month_date)

    # Merge with macro data using a scalar for comparison
    final_data = merged_data.merge(
        macro_sme3_lag1[macro_sme3_lag1['DATE'] == merged_data['end_of_month'].iloc[0]],
        left_on='end_of_month',
        right_on='DATE',
        how='left'
    )

    final_data['UNEMPLOYMENT-RATE'] = final_data.apply(
        lambda row: row['UNEMPLOYMENT-RATE-MALE'] if row['gender'] == Gender.MALE else row['UNEMPLOYMENT-RATE-FEMALE'],
        axis=1
    )

    final_data['DTI'] = ((loan_amount / repayment_term_periods) * 2) / final_data.get('median_earning', 1)

    final_data['MALE'] = (final_data['gender'] == Gender.MALE).astype(int)

    final_output = final_data[['AGE', 'MALE', 'MONEY-SUPPLY-M2', 
                                'CONSUMER-CONFIDENCE', 'EXPORT-PRICES', 
                                'USDTWD', 'TWBLR', 
                                'UNEMPLOYMENT-RATE', 'DTI']]
    print(final_output)
    print(final_output.info())
    return final_output


def predict_application_status(df: pd.DataFrame) -> ApplicationStatus:
    """Predict application status based on prepared data."""

    # Ensure the DataFrame has exactly one row
    if df.shape[0] != 1:
        raise ValueError("Input DataFrame must contain exactly one row.")

    # Specify columns used for prediction
    columns_to_predict = ['AGE', 'MALE', 'MONEY-SUPPLY-M2', 
                          'CONSUMER-CONFIDENCE', 'EXPORT-PRICES', 
                          'USDTWD', 'TWBLR', 'UNEMPLOYMENT-RATE', 'DTI']

    # Extract features from the input DataFrame
    features_to_predict = df[columns_to_predict]

    # Make predictions on the input data using the loaded model
    y_pred_proba = best_model.predict_proba(features_to_predict)[:, 1]

    # Determine application status based on predicted probability
    predicted_status = ApplicationStatus.DENIED if y_pred_proba[0] >= 0.5 else ApplicationStatus.APPROVED

    print("Predicted Probability:", y_pred_proba[0])
    print("Predicted Class:", predicted_status)

    return predicted_status
