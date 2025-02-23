import os
from sqlmodel import SQLModel, create_engine, Session, Field
from typing import Optional
from datetime import datetime

# Database configuration
DBUSER = os.environ.get("DBUSER")
DBPASS = os.environ.get("DBPASS")
DBHOST = os.environ.get("DBHOST")
DBNAME = os.environ.get("DBNAME")
DBPORT = os.environ.get("DBPORT")

# If all environment variables are set, use PostgreSQL, otherwise use SQLite
if all([DBUSER, DBPASS, DBHOST, DBNAME, DBPORT]):
    DATABASE_URL = f"postgresql://{DBUSER}:{DBPASS}@{DBHOST}:{DBPORT}/{DBNAME}"
else:
    # Fallback to SQLite for development
    DATABASE_URL = "sqlite:///./recommendation.db"

engine = create_engine(DATABASE_URL, echo=True)

def init_db():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

# Define your database models
class BaseModel(SQLModel):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.now)

class User(BaseModel, table=True):
    email: str = Field(unique=True, index=True)
    hashed_password: str

class Restaurant(BaseModel, table=True):
    name: str = Field(index=True)
    rating: float = Field(default=0.0)
    price_level: int
    cuisine: str = Field(index=True)
    location: str = Field(index=True)

class Rating(BaseModel, table=True):
    restaurant_id: int = Field(foreign_key="restaurant.id")
    user_id: int = Field(foreign_key="user.id")
    user_rating: float
    review_text: Optional[str] = None

