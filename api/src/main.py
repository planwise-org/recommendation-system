from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import users, places, reviews, recommendations
from .database import engine, Base, init_db, supabase
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
        env = os.environ.get("ENV")
        
        if env == "test":
            # Test environment uses SQLAlchemy with SQLite
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                logger.debug("Database connection test successful")
            init_db()
        else:
            # Local and prod environments use Supabase
            # Test Supabase connection by fetching from user table
            try:
                supabase.table("user").select("*").limit(1).execute()
                logger.debug("Supabase connection test successful")
            except Exception as e:
                logger.error(f"Supabase connection test failed: {str(e)}")
                raise
                
        logger.debug("Database initialization completed")
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
