from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from typing import List
from ..models import Recommendation
from ..schemas.recommendation import RecommendationCreate, RecommendationRead, RecommendationUpdate, RecommendationAlgorithm
from ..database import get_session
from ..services.recommendation import generate_recommendations

router = APIRouter()

@router.post("/", response_model=RecommendationRead, status_code=status.HTTP_201_CREATED)
def create_recommendation(recommendation: RecommendationCreate, db: Session = Depends(get_session)):
    db_recommendation = Recommendation(**recommendation.dict())
    db.add(db_recommendation)
    db.commit()
    db.refresh(db_recommendation)
    return db_recommendation

@router.get("/", response_model=List[RecommendationRead])
def get_recommendations(
    skip: int = 0,
    limit: int = 100,
    user_id: int = None,
    algorithm: str = None,
    db: Session = Depends(get_session)
):
    query = select(Recommendation)
    if user_id:
        query = query.where(Recommendation.user_id == user_id)
    if algorithm:
        query = query.where(Recommendation.algorithm == algorithm)
    recommendations = db.exec(query.offset(skip).limit(limit)).all()
    return recommendations

@router.post("/generate/{user_id}", response_model=List[RecommendationRead])
def generate_user_recommendations(
    user_id: int,
    algorithm: RecommendationAlgorithm = RecommendationAlgorithm.AUTOENCODER,
    db: Session = Depends(get_session)
):
    recommendations = generate_recommendations(db, user_id, algorithm)
    db_recommendations = []
    for rec in recommendations:
        db_rec = Recommendation(
            user_id=rec["user_id"],
            place_id=rec["place_id"],
            algorithm=rec["algorithm"],
            score=rec["score"],
            visited=False,
            reviewed=False
        )
        db.add(db_rec)
        db_recommendations.append(db_rec)
    db.commit()
    for rec in db_recommendations:
        db.refresh(rec)
    return db_recommendations

@router.get("/{recommendation_id}", response_model=RecommendationRead)
def get_recommendation(recommendation_id: int, db: Session = Depends(get_session)):
    recommendation = db.get(Recommendation, recommendation_id)
    if not recommendation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recommendation not found"
        )
    return recommendation

@router.put("/{recommendation_id}", response_model=RecommendationRead)
def update_recommendation(
    recommendation_id: int,
    recommendation_update: RecommendationUpdate,
    db: Session = Depends(get_session)
):
    db_recommendation = db.get(Recommendation, recommendation_id)
    if not db_recommendation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recommendation not found"
        )

    # Update recommendation fields
    for field, value in recommendation_update.dict(exclude_unset=True).items():
        setattr(db_recommendation, field, value)

    db.add(db_recommendation)
    db.commit()
    db.refresh(db_recommendation)
    return db_recommendation

@router.put("/{recommendation_id}/visited", response_model=RecommendationRead)
def mark_recommendation_visited(recommendation_id: int, db: Session = Depends(get_session)):
    db_recommendation = db.get(Recommendation, recommendation_id)
    if not db_recommendation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recommendation not found"
        )

    db_recommendation.visited = True
    db.add(db_recommendation)
    db.commit()
    db.refresh(db_recommendation)
    return db_recommendation

@router.put("/{recommendation_id}/reviewed", response_model=RecommendationRead)
def mark_recommendation_reviewed(recommendation_id: int, db: Session = Depends(get_session)):
    db_recommendation = db.get(Recommendation, recommendation_id)
    if not db_recommendation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recommendation not found"
        )

    db_recommendation.reviewed = True
    db.add(db_recommendation)
    db.commit()
    db.refresh(db_recommendation)
    return db_recommendation

@router.delete("/{recommendation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_recommendation(recommendation_id: int, db: Session = Depends(get_session)):
    recommendation = db.get(Recommendation, recommendation_id)
    if not recommendation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recommendation not found"
        )

    db.delete(recommendation)
    db.commit()
    return None
