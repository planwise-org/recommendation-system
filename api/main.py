from fastapi import FastAPI
from routes.users import router as users_router
from routes.recommendations import router as recommendations_router
from database import init_db
from sklearn.preprocessing import StandardScaler
import joblib
import os

app = FastAPI()



# initialize the db on startup
@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/")
async def read_root():
    return {"message": "Planwise says Hello World!"}

# simple setup for including the router page
app.include_router(users_router, prefix="/users", tags=["Users"])
app.include_router(recommendations_router, prefix="/recommendations", tags=["Recommendations"])

def save_scaler(scaler):
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    # Save the scaler
    joblib.dump(scaler, 'models/scaler.save')

