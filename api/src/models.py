from sqlmodel import Field, SQLModel, Relationship
from datetime import datetime
from typing import Optional, List
from enum import Enum


"""
This file contains the models for the database tables.

We have 7 tables:
    - User
    - Place
    - Plan
    - Route
    - Review
    - Category
    - Preference
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
    username: str = Field(unique=True, index=True)
    hashed_password: str
    full_name: str
    role: UserRole = Field(default=UserRole.USER)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships with cascade delete
    reviews: List["Review"] = Relationship(back_populates="user", sa_relationship_kwargs={"cascade": "all, delete-orphan"})
    recommendations: List["Recommendation"] = Relationship(back_populates="user", sa_relationship_kwargs={"cascade": "all, delete-orphan"})
    preferences: List["Preference"] = Relationship(back_populates="user", sa_relationship_kwargs={"cascade": "all, delete-orphan"})

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

    # Remove the reviews relationship since we're using Google Places IDs
    # reviews: List["Review"] = Relationship(back_populates="place")

    # Relationships
    recommendations: List["Recommendation"] = Relationship(back_populates="place")

    class Config:
        arbitrary_types_allowed = True

class Review(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", nullable=False)
    place_id: int = Field(foreign_key="place.id", nullable=False)
    rating: float = Field(ge=1.0, le=5.0)
    comment: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    user: User = Relationship(back_populates="reviews")

    class Config:
        arbitrary_types_allowed = True

class Recommendation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", nullable=False)
    place_id: int = Field(foreign_key="place.id", nullable=False)
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

class Preference(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", nullable=False)
    category: str = Field(index=True)
    rating: float = Field(ge=0, le=5)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    user: User = Relationship(back_populates="preferences")

    class Config:
        arbitrary_types_allowed = True
