from crud import create_tables
from crud import create_clients
from crud import create_earnings
from crud import create_macroeconomic_data
from crud import get_session


def init_db(session):
    """Populate the database tables."""
    print("Starting database population...")

    create_tables()

    print("Creating clients...")
    create_clients(session)

    print("Creating earnings data...")
    create_earnings(session)

    print("Creating macroeconomic data...")
    create_macroeconomic_data(session)

    print("Finished database population.")


if __name__ == "__main__":
    with get_session() as session:
        init_db(session)
