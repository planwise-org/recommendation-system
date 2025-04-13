from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict
from datetime import datetime
from src.models import UserRole, RecommendationAlgorithm

class UserBase(BaseModel):
    email: EmailStr
    name: str

class UserCreate(UserBase):
    password: str

class UserUpdate(BaseModel):
    name: Optional[str] = None
    initial_preferences: Optional[Dict] = None
    current_preferences: Optional[Dict] = None

class UserResponse(UserBase):
    id: int
    role: UserRole
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class PlaceBase(BaseModel):
    name: str
    description: Optional[str] = None
    latitude: float
    longitude: float
    address: Optional[str] = None
    category: str
    types: List[str]
    price_level: Optional[int] = None
    opening_hours: Optional[Dict] = None
    photos: List[str]

class PlaceCreate(PlaceBase):
    pass

class PlaceUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    rating: Optional[float] = None
    total_ratings: Optional[int] = None
    price_level: Optional[int] = None
    opening_hours: Optional[Dict] = None
    photos: Optional[List[str]] = None

class PlaceResponse(PlaceBase):
    id: int
    rating: Optional[float] = None
    total_ratings: Optional[int] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class ReviewBase(BaseModel):
    rating: float
    comment: Optional[str] = None
    photos: List[str]
    visit_date: Optional[datetime] = None

class ReviewCreate(ReviewBase):
    place_id: int

class ReviewUpdate(BaseModel):
    rating: Optional[float] = None
    comment: Optional[str] = None
    photos: Optional[List[str]] = None
    visit_date: Optional[datetime] = None

class ReviewResponse(ReviewBase):
    id: int
    user_id: int
    place_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class RecommendationBase(BaseModel):
    place_id: int
    algorithm: RecommendationAlgorithm
    score: float

class RecommendationCreate(RecommendationBase):
    pass

class RecommendationResponse(RecommendationBase):
    id: int
    user_id: int
    was_visited: bool
    was_reviewed: bool
    created_at: datetime
    viewed_at: Optional[datetime] = None
    visited_at: Optional[datetime] = None

    class Config:
        from_attributes = True