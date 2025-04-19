from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class RecommendationBase(BaseModel):
    user_id: int
    place_id: int
    algorithm: str
    score: float

class RecommendationCreate(RecommendationBase):
    pass

class RecommendationRead(RecommendationBase):
    id: int
    visited: bool
    reviewed: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class RecommendationUpdate(BaseModel):
    visited: Optional[bool] = None
    reviewed: Optional[bool] = None 