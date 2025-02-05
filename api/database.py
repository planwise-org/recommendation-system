
#how our backend connects to the database:

from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker, declarative_base
import os

DATABASE_URL = "postgresql://alexandrakhreiche:alexa@localhost/travel_db"

engine = create_engine(DATABASE_URL)
# SessionLocal() to handle database transactions (i.e., read/write data)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
#Base = declarative_base() to allow us to define tables as Python classes.
Base = declarative_base()

