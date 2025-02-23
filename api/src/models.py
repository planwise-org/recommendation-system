from sqlmodel import Field, SQLModel


"""
This file contains the models for the database tables.

We have 6 tables:
    - User
    - Place
    - Plan
    - Route
    - Review
    - Category
"""

class User(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    email: str = Field()
    hash_password: str = Field()
    preferences: str = Field()
