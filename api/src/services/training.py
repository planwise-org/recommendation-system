# api/src/services/training.py
from sqlmodel import Session, select
from models import Review, Place
from src.ensemble import EnsembleRecommender
import pandas as pd

# Global model reference
ensemble_model = None

def retrain_ensemble_model(db: Session):
    global ensemble_model

    reviews = db.exec(select(Review)).all()
    if not reviews:
        return

    # Convert to DataFrame
    df = pd.DataFrame([
        {"user": f"user_{r.user_id}", "place_id": r.place_id, "rating": r.rating}
        for r in reviews
    ])

    # Load place metadata
    places = db.exec(select(Place)).all()
    places_df = pd.DataFrame([{
        "place_id": p.id,
        "name": p.name,
        "lat": p.latitude,
        "lng": p.longitude,
        "category": p.place_type.value
    } for p in places])

    ensemble_model = EnsembleRecommender()
    ensemble_model.initialize_models_from_reviews(df, places_df)

    print(" Ensemble model retrained")