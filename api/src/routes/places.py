from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from typing import List
from ..models import Place
from ..schemas.place import PlaceCreate, PlaceRead, PlaceUpdate
from ..database import get_session

router = APIRouter()

@router.post("/", response_model=PlaceRead, status_code=status.HTTP_201_CREATED)
def create_place(place: PlaceCreate, db: Session = Depends(get_session)):
    db_place = Place(**place.dict())
    db.add(db_place)
    db.commit()
    db.refresh(db_place)
    return db_place

@router.get("/", response_model=List[PlaceRead])
def get_places(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_session)
):
    places = db.exec(select(Place).offset(skip).limit(limit)).all()
    return places

@router.get("/{place_id}", response_model=PlaceRead)
def get_place(place_id: int, db: Session = Depends(get_session)):
    place = db.get(Place, place_id)
    if not place:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Place not found"
        )
    return place

@router.put("/{place_id}", response_model=PlaceRead)
def update_place(
    place_id: int,
    place_update: PlaceUpdate,
    db: Session = Depends(get_session)
):
    db_place = db.get(Place, place_id)
    if not db_place:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Place not found"
        )

    # Update place fields
    for field, value in place_update.dict(exclude_unset=True).items():
        setattr(db_place, field, value)

    db.add(db_place)
    db.commit()
    db.refresh(db_place)
    return db_place

@router.delete("/{place_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_place(place_id: int, db: Session = Depends(get_session)):
    place = db.get(Place, place_id)
    if not place:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Place not found"
        )

    db.delete(place)
    db.commit()
    return None
