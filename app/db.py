from sqlmodel import create_engine

DB_HOST = 'database'
DB_PORT = '5432'
DB_USER = 'postgres'
DB_PASS = 'postgres'
DB_NAME = 'sample_db'

DATABASE_URL = f'postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

engine = create_engine(DATABASE_URL, echo=True)
