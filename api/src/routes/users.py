from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from typing import List
from ..models import User, UserRole
from ..database import get_session
from ..schemas.user import UserCreate, UserRead, UserUpdate
from ..services.auth import get_password_hash
from supabase import Client

router = APIRouter()

@router.post("/", response_model=UserRead, status_code=status.HTTP_201_CREATED)
def create_user(user: UserCreate, db: Session | Client = Depends(get_session)):
    # Check if user with email already exists
    if isinstance(db, Client):  # Supabase
        existing_user = db.table("user").select("*").eq("email", user.email).execute()
        if existing_user.data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        # Create new user
        hashed_password = get_password_hash(user.password)
        user_data = {
            "email": user.email,
            "hash_password": hashed_password,  # Note: column name in Supabase
            "name": user.name,  # Changed from full_name to name
            "preferences": "{}"  # Required field in Supabase
        }
        
        try:
            result = db.table("user").insert(user_data).execute()
            return result.data[0]
        except Exception as e:
            print(f"Error creating user: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not create user"
            )
    else:  # SQLite (test environment)
        db_user = db.exec(select(User).where(User.email == user.email)).first()
        if db_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        hashed_password = get_password_hash(user.password)
        db_user = User(
            email=user.email,
            hashed_password=hashed_password,
            full_name=user.full_name,
            role=user.role
        )
        try:
            db.add(db_user)
            db.commit()
            db.refresh(db_user)
            return db_user
        except Exception as e:
            db.rollback()
            print(f"Error creating user: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not create user"
            )

@router.get("/", response_model=List[UserRead])
def get_users(
    skip: int = 0,
    limit: int = 100,
    db: Session | Client = Depends(get_session)
):
    if isinstance(db, Client):  # Supabase
        result = db.table("user").select("*").range(skip, skip + limit).execute()
        return result.data
    else:  # SQLite (test environment)
        users = db.exec(select(User).offset(skip).limit(limit)).all()
        return users

@router.get("/{user_id}", response_model=UserRead)
def get_user(user_id: int, db: Session | Client = Depends(get_session)):
    if isinstance(db, Client):  # Supabase
        result = db.table("user").select("*").eq("id", user_id).execute()
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        return result.data[0]
    else:  # SQLite (test environment)
        user = db.get(User, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        return user

@router.put("/{user_id}", response_model=UserRead)
def update_user(
    user_id: int,
    user_update: UserUpdate,
    db: Session | Client = Depends(get_session)
):
    if isinstance(db, Client):  # Supabase
        # First check if user exists
        result = db.table("user").select("*").eq("id", user_id).execute()
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Prepare update data
        update_data = user_update.dict(exclude_unset=True)
        if "password" in update_data:
            update_data["hash_password"] = get_password_hash(update_data.pop("password"))
        if "full_name" in update_data:
            update_data["name"] = update_data.pop("full_name")

        result = db.table("user").update(update_data).eq("id", user_id).execute()
        return result.data[0]
    else:  # SQLite (test environment)
        db_user = db.get(User, user_id)
        if not db_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        for field, value in user_update.dict(exclude_unset=True).items():
            if field == "password" and value:
                value = get_password_hash(value)
                field = "hashed_password"
            setattr(db_user, field, value)

        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user

@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: int, db: Session | Client = Depends(get_session)):
    if isinstance(db, Client):  # Supabase
        result = db.table("user").delete().eq("id", user_id).execute()
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
    else:  # SQLite (test environment)
        user = db.get(User, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        db.delete(user)
        db.commit()
    return None
