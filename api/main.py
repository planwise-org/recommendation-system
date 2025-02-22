from fastapi import FastAPI
from src.routes.users import users
from database import init_db, get_session





app = FastAPI()



# initialize the db on startup
@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/")
async def read_root():
    return {"message": "Planwise says Hello World!"}

# simple setup for including the router page
app.include_router(users)
