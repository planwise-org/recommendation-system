from sqlmodel import SQLModel, create_engine, Session
from typing import Generator
import os
from dotenv import load_dotenv
import logging
from sqlalchemy.exc import SQLAlchemyError

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get database URL from environment variable or use default
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/planwise_db")
logger.debug(f"Connecting to database at: {DATABASE_URL}")

# Create SQLModel engine with a timeout
engine = create_engine(
    DATABASE_URL,
    echo=True,
    pool_pre_ping=True,
    pool_timeout=30,
    pool_recycle=1800
)

# Create Base class for models
Base = SQLModel

def get_session() -> Generator[Session, None, None]:
    """
    Get a database session.
    """
    logger.debug("Creating new database session")
    try:
        with Session(engine) as session:
            yield session
    except SQLAlchemyError as e:
        logger.error(f"Database session error: {str(e)}")
        raise

def init_db():
    """
    Initialize the database by creating all tables if they don't exist.
    """
    logger.debug("Initializing database tables")
    try:
        SQLModel.metadata.create_all(engine)
        logger.debug("Database tables initialized successfully")
    except SQLAlchemyError as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

def drop_all_tables():
    """
    Drop all tables in the database.
    """
    logger.debug("Dropping all tables")
    try:
        SQLModel.metadata.drop_all(engine)
        logger.debug("All tables dropped successfully")
    except SQLAlchemyError as e:
        logger.error(f"Error dropping tables: {str(e)}")
        raise

def recreate_tables():
    """
    Drop all tables and recreate them.
    """
    logger.debug("Starting table recreation")
    try:
        drop_all_tables()
        init_db()
        logger.debug("Table recreation completed")
    except SQLAlchemyError as e:
        logger.error(f"Error during table recreation: {str(e)}")
        raise
