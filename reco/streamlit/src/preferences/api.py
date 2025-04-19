# api.py
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
from typing import List


from ..models import (
    Base,  # ← this Base!
    User, UserCreate, UserResponse, UserLogin,
    Preference, PreferenceCreate, PreferenceResponse,
    Review, ReviewCreate, ReviewResponse
)
from ..database import get_db, engine

from ..auth import (
    get_password_hash, verify_password, create_access_token,
    get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES
)
from .preference_extractor import pearl_extract_preferences_single

from pydantic import BaseModel

class TextInput(BaseModel):
    text: str

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Enhanced Recommendation System API",
    description="API for user management, preferences, and reviews",
    version="1.0"
)

# Auth endpoints
@app.post("/token")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/users/", response_model=UserResponse)
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# Preference endpoints
@app.post("/extract-preferences")
async def extract_preferences(
    input: TextInput,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    text = input.text

    extracted_prefs = pearl_extract_preferences_single(text)
    
    for category, rating in extracted_prefs.items():
        db_pref = Preference(
            user_id=current_user.id,
            category=category,
            rating=rating
        )
        db.add(db_pref)
    db.commit()
    
    return {"preferences": extracted_prefs}

@app.get("/preferences/", response_model=List[PreferenceResponse])
async def get_user_preferences(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return db.query(Preference).filter(Preference.user_id == current_user.id).all()

@app.post("/preferences/", response_model=PreferenceResponse)
async def create_preference(
    preference: PreferenceCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    db_preference = Preference(**preference.dict(), user_id=current_user.id)
    db.add(db_preference)
    db.commit()
    db.refresh(db_preference)
    return db_preference

@app.delete("/preferences/{category}")
async def delete_preference(
    category: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    preference = db.query(Preference).filter(
        Preference.user_id == current_user.id,
        Preference.category == category
    ).first()
    
    if preference:
        db.delete(preference)
        db.commit()
        return {"message": f"Preference for {category} deleted"}
    return {"message": f"No preference found for {category}"}

# Review endpoints
@app.post("/reviews/", response_model=ReviewResponse)
async def create_review(
    review: ReviewCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    db_review = Review(**review.dict(), user_id=current_user.id)
    db.add(db_review)
    db.commit()
    db.refresh(db_review)
    return db_review

@app.get("/reviews/", response_model=List[ReviewResponse])
async def get_user_reviews(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return db.query(Review).filter(Review.user_id == current_user.id).all()

@app.get("/reviews/{place_id}", response_model=List[ReviewResponse])
async def get_place_reviews(
    place_id: str,
    db: Session = Depends(get_db)
):
    return db.query(Review).filter(Review.place_id == place_id).all()

@app.get("/reviews/user/{place_id}")
async def get_user_place_review(
    place_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    review = db.query(Review).filter(
        Review.user_id == current_user.id,
        Review.place_id == place_id
    ).order_by(Review.id.desc()).first()
    
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

@app.get("/users/{username}/exists")
async def check_user_exists(username: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"exists": True}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
