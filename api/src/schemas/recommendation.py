from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from enum import Enum


class RecommendationAlgorithm(str, Enum):
    AUTOENCODER = "autoencoder"
    SVD = "svd"
    TRANSFER_LEARNING = "transfer_learning"

class RecommendationBase(BaseModel):
    user_id: int
    place_id: int
    algorithm: RecommendationAlgorithm
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
