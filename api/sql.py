import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from api.models import Base
from sqlite3 import Connection as Sqlite3Connection
from sqlalchemy.orm import sessionmaker

load_dotenv()

DB_NAME = os.getenv("DB_NAME")
engine = create_engine(f"sqlite:///{DB_NAME}", echo=True)


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    if isinstance(dbapi_connection, Sqlite3Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.close()


if __name__ == "__main__":
    Base.metadata.create_all(engine)

SessionLocal = sessionmaker(
    bind=engine, expire_on_commit=False, autocommit=False)
