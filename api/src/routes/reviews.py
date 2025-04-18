from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from typing import List
from ..models import Review
from ..schemas.review import ReviewCreate, ReviewRead, ReviewUpdate
from ..database import get_session

router = APIRouter()

@router.post("/", response_model=ReviewRead, status_code=status.HTTP_201_CREATED)
def create_review(review: ReviewCreate, db: Session = Depends(get_session)):
    db_review = Review(**review.dict())
    db.add(db_review)
    db.commit()
    db.refresh(db_review)
    return db_review

@router.get("/", response_model=List[ReviewRead])
def get_reviews(
    skip: int = 0,
    limit: int = 100,
    user_id: int = None,
    place_id: int = None,
    db: Session = Depends(get_session)
):
    query = select(Review)
    if user_id:
        query = query.where(Review.user_id == user_id)
    if place_id:
        query = query.where(Review.place_id == place_id)
    reviews = db.exec(query.offset(skip).limit(limit)).all()
    return reviews

@router.get("/{review_id}", response_model=ReviewRead)
def get_review(review_id: int, db: Session = Depends(get_session)):
    review = db.get(Review, review_id)
    if not review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Review not found"
        )
    return review

@router.put("/{review_id}", response_model=ReviewRead)
def update_review(
    review_id: int,
    review_update: ReviewUpdate,
    db: Session = Depends(get_session)
):
    db_review = db.get(Review, review_id)
    if not db_review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Review not found"
        )
    
    # Update review fields
    for field, value in review_update.dict(exclude_unset=True).items():
        setattr(db_review, field, value)
    
    db.add(db_review)
    db.commit()
    db.refresh(db_review)
    return db_review

@router.delete("/{review_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_review(review_id: int, db: Session = Depends(get_session)):
    review = db.get(Review, review_id)
    if not review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Review not found"
        )
    
    db.delete(review)
    db.commit()
    return None 