import os
from sqlmodel import create_engine, SQLModel, Session
from supabase import create_client, Client



# when on production, use the supabase client
# when on local, use the local database connection created on the docker-compose file
# when on test, use in-memory SQLite database

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

def init_db():
    if os.environ.get("ENV") in ["local", "test"]:
        SQLModel.metadata.create_all(engine)

def get_db():
    if os.environ.get("ENV") in ["test", "local"]:
        with Session(engine) as session:
            yield session
    elif os.environ.get("ENV") == "prod":
        yield supabase
