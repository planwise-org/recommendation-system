from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session
from ..models import User
from services.auth import hash_password
from schemas import UserCreate
from database import get_db


users = APIRouter()


@users.get("/")
async def read_users():
    return {"Hello": "Users"}


@users.post("/register/")
def register(user: UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_pw = hash_password(user.password)
    new_user = User(email=user.email, hashed_password=hashed_pw)

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": "User created"}
