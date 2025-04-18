from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import users, places, reviews, recommendations
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
        if os.environ.get("ENV") == "local":
            from .database import engine
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                logger.debug("Database connection test successful")

        # Initialize tables if they don't exist
        init_db()
        logger.debug("Database initialization completed")
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

# Include routers
app.include_router(users.router, prefix="/api/users", tags=["Users"])
app.include_router(places.router, prefix="/api/places", tags=["Places"])
app.include_router(reviews.router, prefix="/api/reviews", tags=["Reviews"])
app.include_router(recommendations.router, prefix="/api/recommendations", tags=["Recommendations"])

@app.get("/")
async def root():
    return {"message": "Welcome to Planwise API"}
