from fastapi import FastAPI
from api.routes import users

app = FastAPI()

app.include_router(users.router, prefix="/users", tags=["Users"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Travel Recommendation API!"}
