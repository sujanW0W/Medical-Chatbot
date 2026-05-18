import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

load_dotenv()

db = os.getenv("POSTGRES_DB")

user = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")
host = os.getenv("POSTGRES_HOST")
port = os.getenv("POSTGRES_PORT")

DATABASE_URL = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"

engine = create_engine(
    DATABASE_URL, echo=True)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# SQLite Foreign key
# @event.listens_for(Engine, "connect")
# def set_sqlite_pragma(dbapi_connection, connection_record):
#     if isinstance(dbapi_connection, Sqlite3Connection):
#         cursor = dbapi_connection.cursor()
#         cursor.execute("PRAGMA foreign_keys = ON")
#         cursor.close()
