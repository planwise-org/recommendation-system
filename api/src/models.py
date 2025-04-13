from sqlmodel import Field, SQLModel, Relationship
from datetime import datetime
from typing import Optional, Dict, List
from enum import Enum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import JSON, Enum as SQLAlchemyEnum


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

class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"

class PlaceType(str, Enum):
    RESTAURANT = "restaurant"
    CAFE = "cafe"
    BAR = "bar"
    CLUB = "club"
    SHOPPING = "shopping"
    ATTRACTION = "attraction"
    PARK = "park"
    MUSEUM = "museum"
    THEATER = "theater"
    OTHER = "other"

class RecommendationAlgorithm(str, Enum):
    AUTOENCODER = "autoencoder"
    SVD = "svd"
    TRANSFER_LEARNING = "transfer_learning"

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(unique=True, index=True)
    hashed_password: str
    full_name: str
    role: UserRole = Field(default=UserRole.USER)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    reviews: List["Review"] = Relationship(back_populates="user")
    recommendations: List["Recommendation"] = Relationship(back_populates="user")

    class Config:
        arbitrary_types_allowed = True

class Place(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    description: str
    address: str
    latitude: float
    longitude: float
    place_type: PlaceType
    rating: float = Field(default=0.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    reviews: List["Review"] = Relationship(back_populates="place")
    recommendations: List["Recommendation"] = Relationship(back_populates="place")

    class Config:
        arbitrary_types_allowed = True

class Review(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    place_id: int = Field(foreign_key="place.id")
    rating: float = Field(ge=1, le=5)
    comment: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    user: User = Relationship(back_populates="reviews")
    place: Place = Relationship(back_populates="reviews")

    class Config:
        arbitrary_types_allowed = True

class Recommendation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    place_id: int = Field(foreign_key="place.id")
    algorithm: RecommendationAlgorithm
    score: float = Field(ge=0, le=1)
    visited: bool = Field(default=False)
    reviewed: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    user: User = Relationship(back_populates="recommendations")
    place: Place = Relationship(back_populates="recommendations")

    class Config:
        arbitrary_types_allowed = True
