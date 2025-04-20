from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from typing import List, Dict
from pydantic import BaseModel
from ..models import Preference
from ..database import get_session
from ..schemas.preference import PreferenceCreate, PreferenceRead, PreferenceUpdate
from ..services.auth import get_current_user
import spacy
import textblob

router = APIRouter()

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define categories and their keywords
CATEGORIES = {
    "resorts": ["resort", "spa", "luxury"],
    "burger/pizza shops": ["burger", "pizza", "fast food"],
    "hotels/other lodgings": ["hotel", "hostel", "lodging", "accommodation"],
    "juice bars": ["juice", "smoothie", "healthy drinks"],
    "beauty & spas": ["beauty", "spa", "massage", "wellness"],
    "gardens": ["garden", "botanical", "flowers"],
    "amusement parks": ["amusement park", "theme park", "rides"],
    "farmer market": ["farmer market", "fresh produce", "local food"],
    "market": ["market", "shopping", "vendors"],
    "music halls": ["music", "concert", "live performance"],
    "nature": ["nature", "outdoors", "hiking", "wilderness"],
    "tourist attractions": ["tourist", "attraction", "sightseeing"],
    "beaches": ["beach", "coast", "seaside"],
    "parks": ["park", "recreation", "outdoor space"],
    "theatres": ["theatre", "theater", "performance", "play"],
    "museums": ["museum", "art", "history", "exhibition"],
    "malls": ["mall", "shopping center", "retail"],
    "restaurants": ["restaurant", "dining", "food"],
    "pubs/bars": ["pub", "bar", "drinks", "nightlife"],
    "local services": ["local", "service", "community"],
    "art galleries": ["art", "gallery", "exhibition"],
    "dance clubs": ["dance", "club", "nightclub"],
    "swimming pools": ["swimming", "pool", "aquatic"],
    "bakeries": ["bakery", "pastry", "bread"],
    "cafes": ["cafe", "coffee", "tea"],
    "view points": ["view", "lookout", "scenic"],
    "monuments": ["monument", "memorial", "historic"],
    "zoo": ["zoo", "animal", "wildlife"],
    "supermarket": ["supermarket", "grocery", "store"]
}

class TextInput(BaseModel):
    text: str

class PreferenceResponse(BaseModel):
    preferences: Dict[str, float]

@router.post("/extract-preferences", response_model=PreferenceResponse)
def extract_preferences(
    text_input: TextInput,
    current_user = Depends(get_current_user)
) -> PreferenceResponse:
    """Extract preferences from text and return a dictionary of category ratings."""
    text = text_input.text.lower()
    doc = nlp(text)

    # Initialize sentiment analyzer
    blob = textblob.TextBlob(text)
    overall_sentiment = blob.sentiment.polarity

    # Default rating based on overall sentiment
    default_rating = 3.0 + (overall_sentiment * 2)  # Scale to 1-5 range

    preferences = {}

    # Extract preferences based on keyword matches and local sentiment
    for category, keywords in CATEGORIES.items():
        # Check for keyword matches
        matches = []
        for keyword in keywords:
            if keyword in text:
                # Find the sentence containing the keyword
                for sent in doc.sents:
                    if keyword in sent.text.lower():
                        matches.append(sent.text)

        if matches:
            # Calculate sentiment for matched sentences
            local_sentiment = sum(textblob.TextBlob(match).sentiment.polarity for match in matches) / len(matches)
            # Convert sentiment to rating (1-5 scale)
            rating = 3.0 + (local_sentiment * 2)
            # Ensure rating is within bounds
            rating = max(1.0, min(5.0, rating))
            preferences[category] = rating
    return PreferenceResponse(preferences=preferences)

@router.post("/", response_model=PreferenceRead)
def create_preference(
    preference: PreferenceCreate,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    # Check if preference already exists
    existing = db.exec(
        select(Preference).where(
            Preference.user_id == current_user.id,
            Preference.category == preference.category
        )
    ).first()

    if existing:
        # Update existing preference
        existing.rating = preference.rating
        db.add(existing)
        db.commit()
        db.refresh(existing)
        return existing

    # Create new preference
    db_preference = Preference(
        user_id=current_user.id,
        category=preference.category,
        rating=preference.rating
    )
    db.add(db_preference)
    db.commit()
    db.refresh(db_preference)
    return db_preference

@router.get("/", response_model=List[PreferenceRead])
def get_preferences(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    preferences = db.exec(
        select(Preference)
        .where(Preference.user_id == current_user.id)
    ).all()
    return preferences

@router.get("/{category}", response_model=PreferenceRead)
def get_preference(
    category: str,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    preference = db.exec(
        select(Preference)
        .where(
            Preference.user_id == current_user.id,
            Preference.category == category
        )
    ).first()
    if not preference:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Preference not found"
        )
    return preference

@router.put("/{category}", response_model=PreferenceRead)
def update_preference(
    category: str,
    preference_update: PreferenceUpdate,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    preference = db.exec(
        select(Preference)
        .where(
            Preference.user_id == current_user.id,
            Preference.category == category
        )
    ).first()
    if not preference:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Preference not found"
        )

    preference.rating = preference_update.rating
    db.add(preference)
    db.commit()
    db.refresh(preference)
    return preference

@router.delete("/{category}", status_code=status.HTTP_204_NO_CONTENT)
def delete_preference(
    category: str,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    preference = db.exec(
        select(Preference)
        .where(
            Preference.user_id == current_user.id,
            Preference.category == category
        )
    ).first()
    if not preference:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Preference not found"
        )

    db.delete(preference)
    db.commit()
    return None
