from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from database import get_session, User
from services.auth import hash_password
from pydantic import BaseModel

# Create the router instance
router = APIRouter()

class UserCreate(BaseModel):
    email: str
    password: str

@router.post("/register/")
def register(user: UserCreate, session: Session = Depends(get_session)):
    # Check if user exists
    statement = select(User).where(User.email == user.email)
    existing_user = session.exec(statement).first()
    
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_pw = hash_password(user.password)
    new_user = User(email=user.email, hashed_password=hashed_pw)
    
    session.add(new_user)
    session.commit()
    session.refresh(new_user)
    
    return {"message": "User created"}

@router.get("/")
async def read_users(session: Session = Depends(get_session)):
    return {"message": "User route works!"}
