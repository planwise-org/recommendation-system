from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import joblib
import os
from sklearn.preprocessing import StandardScaler
from database import get_session, Restaurant, Rating  # Changed from api.database
from sqlmodel import Session, select
# from ml.predict import get_predictions

router = APIRouter()

class UserPreferences(BaseModel):
    price_range: Optional[int] = None  # 1-4 for price level
    rating_minimum: Optional[float] = None  # e.g. 4.0
    cuisine_type: Optional[str] = None  # e.g. "Italian"
    location: Optional[str] = None  # e.g. "New York"
    distance_willing: Optional[float] = None  # in miles/km

class RestaurantRating(BaseModel):
    restaurant_id: int
    user_rating: float  # 1-5 stars
    review_text: Optional[str] = None

@router.post("/recommend")
async def get_recommendations(preferences: UserPreferences, session: Session = Depends(get_session)):
    try:
        # Load your trained model and scaler
        predictions = get_predictions(preferences.__dict__)
        
        # Query restaurants from database based on preferences
        query = select(Restaurant)
        if preferences.cuisine_type:
            query = query.where(Restaurant.cuisine == preferences.cuisine_type)
        if preferences.rating_minimum:
            query = query.where(Restaurant.rating >= preferences.rating_minimum)
        if preferences.price_range:
            query = query.where(Restaurant.price_level <= preferences.price_range)
            
        restaurants = session.exec(query).all()
        
        # Process and return recommendations
        recommendations = generate_recommendations(preferences, predictions, restaurants)
        
        return {
            "status": "success",
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rate")
async def rate_restaurant(rating: RestaurantRating, session: Session = Depends(get_session)):
    try:
        # Create new rating in database
        db_rating = Rating(
            restaurant_id=rating.restaurant_id,
            user_rating=rating.user_rating,
            review_text=rating.review_text
        )
        session.add(db_rating)
        session.commit()
        
        return {
            "status": "success",
            "message": "Rating submitted successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_recommendations(preferences, predictions, restaurants):
    # Convert restaurants to list of dictionaries
    return [
        {
            "restaurant_id": restaurant.id,
            "name": restaurant.name,
            "rating": restaurant.rating,
            "price_level": restaurant.price_level,
            "cuisine": restaurant.cuisine,
            "location": restaurant.location,
            "match_score": float(predictions[get_category_index(restaurant.cuisine)])
        }
        for restaurant in restaurants
    ]

def get_category_index(cuisine):
    # Map cuisine to the correct index in predictions array
    categories = [
        "resorts", "burger/pizza shops", "hotels/other lodgings",
        # ... add all categories in the same order as training
    ]
    return categories.index(cuisine)

@router.get("/")
async def get_recommendations(session: Session = Depends(get_session)):
    return {"message": "Recommendations route works!"} 