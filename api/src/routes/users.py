from fastapi import APIRouter


users = APIRouter(
    prefix="/users",
    tags=["users"]
)


@users.get("/")
async def read_users():
    return {"Hello": "Users"}
