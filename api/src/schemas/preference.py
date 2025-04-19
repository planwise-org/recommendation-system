from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class PreferenceBase(BaseModel):
    category: str
    rating: float = Field(ge=0, le=5)

class PreferenceCreate(PreferenceBase):
    pass

class PreferenceUpdate(BaseModel):
    rating: float = Field(ge=0, le=5)

class PreferenceRead(PreferenceBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True 