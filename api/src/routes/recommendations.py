from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from typing import List
from ..models import Recommendation
from ..schemas.recommendation import RecommendationCreate, RecommendationRead, RecommendationUpdate
from ..database import get_session
from ..services.recommendation import generate_recommendations
from ..services.training import ensemble_model
from ..models import User

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
    algorithm: str = "autoencoder",
    db: Session = Depends(get_session)
):
    if algorithm == "ensemble":
        if ensemble_model is None:
            raise HTTPException(status_code=503, detail="Ensemble model not trained yet")

        user = db.get(User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # replace with real user data if available
        user_lat = 40.4168
        user_lon = -3.7038
        user_prefs = {cat: 3.0 for cat in ensemble_model.category_to_place_types.keys()}
        predicted_ratings = user_prefs

        recommendations = ensemble_model.get_recommendations(
            user_lat=user_lat,
            user_lon=user_lon,
            user_prefs=user_prefs,
            predicted_ratings_dict=predicted_ratings,
            num_recs=5
        )

        db_recommendations = []
        for rec in recommendations:
            db_rec = Recommendation(
                user_id=user_id,
                place_id=rec.get("place_id", rec.get("name")),
                algorithm="ensemble",
                score=rec.get("ensemble_score", 0.0),
                visited=False,
                reviewed=False
            )
            db.add(db_rec)
            db_recommendations.append(db_rec)
        db.commit()
        for rec in db_recommendations:
            db.refresh(rec)
        return db_recommendations

    # Otherwise fallback to default 
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