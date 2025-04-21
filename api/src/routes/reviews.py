from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from typing import List
from ..models import Review
from ..schemas.review import ReviewCreate, ReviewRead, ReviewUpdate
from ..database import get_session
from ..services.auth import get_current_user
from datetime import datetime

router = APIRouter()

@router.post("/", response_model=ReviewRead, status_code=status.HTTP_201_CREATED)
def create_review(
    review: ReviewCreate,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    # Check if user already has a review for this place
    existing_review = db.exec(
        select(Review)
        .where(Review.user_id == current_user.id)
        .where(Review.place_id == review.place_id)
    ).first()

    if existing_review:
        # Update existing review
        existing_review.rating = review.rating
        existing_review.comment = review.comment
        existing_review.updated_at = datetime.utcnow()
        db.add(existing_review)
        db.commit()
        db.refresh(existing_review)
        return existing_review

    # Create new review if none exists
    db_review = Review(
        user_id=current_user.id,
        place_id=review.place_id,
        rating=review.rating,
        comment=review.comment
    )
    db.add(db_review)
    db.commit()
    db.refresh(db_review)
    return db_review

@router.get("/", response_model=List[ReviewRead])
def get_reviews(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    query = select(Review).where(Review.user_id == current_user.id)
    reviews = db.exec(query).all()
    return reviews

@router.get("/user/{place_id}")
def get_user_place_review(
    place_id: str,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    review = db.exec(
        select(Review)
        .where(Review.user_id == current_user.id)
        .where(Review.place_id == place_id)
        .order_by(Review.id.desc())
    ).first()
    
    if review:
        return {
            "rating": review.rating,
            "comment": review.comment,
            "submitted": True
        }
    return {
        "rating": 3.0,
        "comment": "",
        "submitted": False
    }

@router.get("/{place_id}", response_model=List[ReviewRead])
def get_place_reviews(
    place_id: str,
    db: Session = Depends(get_session)
):
    query = select(Review).where(Review.place_id == place_id)
    reviews = db.exec(query).all()
    return reviews

@router.put("/{review_id}", response_model=ReviewRead)
def update_review(
    review_id: int,
    review_update: ReviewUpdate,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    db_review = db.get(Review, review_id)
    if not db_review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Review not found"
        )
    
    if db_review.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this review"
        )
    
    # Update review fields
    for field, value in review_update.dict(exclude_unset=True).items():
        setattr(db_review, field, value)
    
    db.add(db_review)
    db.commit()
    db.refresh(db_review)
    return db_review

@router.delete("/{review_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_review(
    review_id: int,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    review = db.get(Review, review_id)
    if not review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Review not found"
        )
    
    if review.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this review"
        )
    
    db.delete(review)
    db.commit()
    return None 