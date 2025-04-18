from pydantic import BaseModel, confloat
from typing import Optional
from datetime import datetime

class ReviewBase(BaseModel):
    rating: confloat(ge=1, le=5)
    comment: str

class ReviewCreate(ReviewBase):
    user_id: int
    place_id: int

class ReviewRead(ReviewBase):
    id: int
    user_id: int
    place_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class ReviewUpdate(BaseModel):
    rating: Optional[confloat(ge=1, le=5)] = None
    comment: Optional[str] = None 