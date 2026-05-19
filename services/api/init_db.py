from api.models import Base
from api.database import engine


def init_database():
    Base.metadata.create_all(bind=engine)
    print("Tables are created successfully.")


if __name__ == "__main__":
    init_database()
