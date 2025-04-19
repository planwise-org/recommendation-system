from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session
from datetime import timedelta
from .database import engine, get_session
from .models import User
from .schemas.user import UserRead
from .services.auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from .routes import users, places, reviews, recommendations, preferences
from .database import init_db
import logging
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Planwise API",
    description="API for the Planwise recommendation system",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    logger.debug("Starting up the application")
    try:
        # Test database connection first
        from .database import engine
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            logger.info("Database connection test successful")

        # Initialize tables if they don't exist
        from .database import init_db
        init_db()
        logger.debug("Database tables initialized successfully")
    except SQLAlchemyError as e:
        logger.error(f"Database error during startup: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during startup: {str(e)}")
        raise

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/token")
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_session)
):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
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

@app.get("/api/users/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

# Include routers
app.include_router(users.router, prefix="/api/users", tags=["Users"])
app.include_router(places.router, prefix="/api/places", tags=["Places"])
app.include_router(reviews.router, prefix="/api/reviews", tags=["Reviews"])
app.include_router(recommendations.router, prefix="/api/recommendations", tags=["Recommendations"])
app.include_router(preferences.router, prefix="/api/preferences", tags=["Preferences"])

@app.get("/")
async def root():
    return {"message": "Welcome to Planwise API"}
