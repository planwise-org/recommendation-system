from sqlmodel import SQLModel, create_engine, Session
from typing import Generator
import os
from dotenv import load_dotenv
import logging
from sqlalchemy.exc import SQLAlchemyError
from supabase import create_client, Client

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

print("ENV: ", os.environ.get("ENV"))

if os.environ.get("ENV") == "test":
    DATABASE_URL = "sqlite:///:memory:"
    engine = create_engine(DATABASE_URL)
    supabase = None

else:  # Both local and prod use Supabase now
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_SERVICE_KEY")  # Use service key for both environments
    supabase: Client = create_client(url, key)
    logger.debug("Connected to Supabase client")
    engine = None  # We don't need SQLAlchemy engine for Supabase

# Create Base class for models
Base = SQLModel

def get_session() -> Generator[Session | Client, None, None]:
    """
    Get a database session.
    For test environment: returns SQLAlchemy session
    For local and prod: returns Supabase client
    """
    logger.debug("Creating new database session")
    if os.environ.get("ENV") == "test":
        with Session(engine) as session:
            yield session
    else:  # Both local and prod use Supabase
        yield supabase

def init_db():
    """
    Initialize the database by creating all tables if they don't exist.
    Only needed for test environment now.
    """
    logger.debug("Initializing database tables")
    if os.environ.get("ENV") == "test":
        try:
            SQLModel.metadata.create_all(engine)
            logger.debug("Database tables initialized successfully")
        except SQLAlchemyError as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise

def drop_all_tables():
    """
    Drop all tables in the database.
    Only needed for test environment now.
    """
    logger.debug("Dropping all tables")
    if os.environ.get("ENV") == "test":
        try:
            SQLModel.metadata.drop_all(engine)
            logger.debug("All tables dropped successfully")
        except SQLAlchemyError as e:
            logger.error(f"Error dropping tables: {str(e)}")
            raise

def recreate_tables():
    """
    Drop all tables and recreate them.
    Only needed for test environment now.
    """
    logger.debug("Starting table recreation")
    if os.environ.get("ENV") == "test":
        try:
            drop_all_tables()
            init_db()
            logger.debug("Table recreation completed")
        except SQLAlchemyError as e:
            logger.error(f"Error during table recreation: {str(e)}")
            raise
