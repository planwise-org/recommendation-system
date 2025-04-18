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


# when on production, use the supabase client
# when on local, use the local database connection created on the docker-compose file
# when on test, use in-memory SQLite database


# Load environment variables
load_dotenv()

if os.environ.get("ENV") == "test":
    DATABASE_URL = "sqlite:///:memory:"
    engine = create_engine(DATABASE_URL)

elif os.environ.get("ENV") == "local":
    # Creates a connection to the database according to the env variables
    DBUSER = os.environ.get("DBUSER")
    DBPASS = os.environ.get("DBPASS")
    DBHOST = os.environ.get("DBHOST")
    DBNAME = os.environ.get("DBNAME")
    DBPORT = os.environ.get("DBPORT")
    DATABASE_URL = f"postgresql://{DBUSER}:{DBPASS}@{DBHOST}:{DBPORT}/{DBNAME}"
    engine = create_engine(DATABASE_URL)


elif os.environ.get("ENV") == "prod":
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_KEY")
    supabase: Client = create_client(url, key)

# Get database URL from environment variable or use default
# DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/planwise_db") # hardcoded

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
    if os.environ.get("ENV") in ["test", "local"]:
        with Session(engine) as session:
            yield session
    elif os.environ.get("ENV") == "prod":
        yield supabase
    else:
        raise ValueError("Invalid environment")


def init_db():
    """
    Initialize the database by creating all tables if they don't exist.
    """
    logger.debug("Initializing database tables")
    if os.environ.get("ENV") in ["test", "local"]:
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
