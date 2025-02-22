from fastapi import FastAPI
from .routes import users

app = FastAPI()



# initialize the db on startup
@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/")
async def read_root():
    return {"message": "Planwise says Hello World!"}

# simple setup for including the router page
app.include_router(users, prefix="/users", tags=["Users"])

