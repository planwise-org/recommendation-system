
#defining the 'users' table in the database 
from sqlalchemy import Column, Integer, String
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    preferences = Column(String, nullable=True)  # JSON format to store user interests


#When a user registers, their data is stored in this table.
#When logging in, the system retrieves the user’s password from this table.