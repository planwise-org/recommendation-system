from fastapi import FastAPI
from src.routes import base

app = FastAPI()

app.include_router(base.router)
