import os
from sqlmodel import create_engine, SQLModel, Session


# initializes the database depending on the environment



DBUSER = os.environ.get("DBUSER")
DBPASS = os.environ.get("DBPASS")
DBHOST = os.environ.get("DBHOST")
DBNAME = os.environ.get("DBNAME")
DBPORT = os.environ.get("DBPORT")


DATABASE_URL = f"postgresql://{DBUSER}:{DBPASS}@{DBHOST}:{DBPORT}/{DBNAME}"

engine = create_engine(DATABASE_URL)


def init_db():
    SQLModel.metadata.create_all(engine)


def get_db():
    with Session(engine) as session:
        yield session

