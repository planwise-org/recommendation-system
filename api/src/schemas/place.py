from pydantic import BaseModel, confloat
from typing import Optional
from datetime import datetime
from ..models import PlaceType

class PlaceBase(BaseModel):
    name: str
    description: str
    address: str
    latitude: confloat(ge=-90, le=90)
    longitude: confloat(ge=-180, le=180)
    place_type: PlaceType

class PlaceCreate(PlaceBase):
    pass

class PlaceRead(PlaceBase):
    id: int
    rating: float
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class PlaceUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    address: Optional[str] = None
    latitude: Optional[confloat(ge=-90, le=90)] = None
    longitude: Optional[confloat(ge=-180, le=180)] = None
    place_type: Optional[PlaceType] = None
    rating: Optional[float] = None 