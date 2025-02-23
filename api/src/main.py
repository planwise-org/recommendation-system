from fastapi import FastAPI
from src.routes.users import users
from src.database import init_db
import os

app = FastAPI()


# create the db on startup
@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/")
async def read_root():
    return {"message": "Planwise says Hello World!"}

# simple setup for including the router page
app.include_router(users, prefix="/api/users", tags=["Users"])
