from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from crud import create_loan_application, get_all_macroeconomic_data, get_all_macroeconomic_data_sme3_lag1, get_all_earnings, get_client_by_email, get_client_by_id, get_session, data_preparation, predict_application_status, load_model
from main import init_db
from validation import LoanApplicationCreate, MacroeconomicDataResponse, EarningsResponse, ClientResponse
from models import LoanApplication
from datetime import datetime
from typing import List, Optional
import logging


logging.basicConfig(level=logging.INFO)

app = FastAPI()


@app.on_event("startup")
def startup_event():
    """Event handler for application startup."""
    print("Starting up the FastAPI application...")

    with get_session() as session:
        init_db(session)

    load_model()


@app.on_event("shutdown")
def shutdown_event():
    """Event handler for application shutdown."""
    print("Shutting down the FastAPI application...")


@app.get("/")
async def read_root():
    return {"message": "Welcome to the API!"}


@app.post("/loan-application", response_model=LoanApplication)
def add_loan_application(
    loan_application: LoanApplicationCreate
):
    """Create and add a new loan application to the database using client's email."""

    client = get_client_by_email(loan_application.client_email)

    if not client:
        logging.error(f"Client not found for email {loan_application.client_email}")
        raise HTTPException(status_code=404, detail="Client not found.")

    logging.info(f"Client found: {client.id} for email {loan_application.client_email}")

    # Pass application_date to data_preparation
    prediction_data = data_preparation(
        client_email=loan_application.client_email,
        loan_amount=loan_application.loan_amount,
        application_date=loan_application.application_date,
        repayment_term_periods=loan_application.repayment_term_periods
    )

    predicted_status = predict_application_status(prediction_data)

    loan_app = LoanApplication(
        client_id=client.id,
        loan_amount=loan_application.loan_amount,
        application_date=loan_application.application_date,
        application_status=predicted_status.value,
        repayment_term_periods=loan_application.repayment_term_periods
    )

    try:
        created_loan_app = create_loan_application(loan_app)
        logging.info(f"Loan application created successfully: {created_loan_app}")
        return created_loan_app
    except Exception as e:
        logging.error(f"Error creating loan application: {e}")
        raise HTTPException(status_code=400, detail="Failed to create loan application.")


@app.get("/macroeconomic-data", response_model=List[MacroeconomicDataResponse])
def get_macroeconomic_data():
    """Retrieve all macroeconomic data records."""
    try:
        macro = get_all_macroeconomic_data()
        return macro
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/macroeconomic-data-sme3-lag1", response_model=List[MacroeconomicDataResponse])
def get_macroeconomic_data_sme3_lag1():
    """Retrieve macroeconomic data with a 3-month rolling average and a 1-month lag."""
    try:
        return get_all_macroeconomic_data_sme3_lag1()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/earnings", response_model=List[EarningsResponse])
def get_earnings():
    """Retrieve all current earnings records."""
    try:
        earnings_records = get_all_earnings()
        return earnings_records
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/client/email", response_model=ClientResponse)
def client_by_email(email: str = Query(...)):
    """Retrieve a client by their email address."""
    try:
        return get_client_by_email(email)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/client/{id}", response_model=ClientResponse)
def client_by_id(id: int):
    """Retrieve a client by their ID."""
    try:
        return get_client_by_id(id)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
