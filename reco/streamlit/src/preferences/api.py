# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Import the single-input extraction function from your module.
from preference_extractor import pearl_extract_preferences_single

# Define the request model.
class PreferenceInput(BaseModel):
    input_text: str
    # Optionally, you can allow the client to set a different weight for exemplar preferences.
    exemplar_weight: float = 0.1

# Define the response model.
class PreferenceOutput(BaseModel):
    preferences: dict

# Create the FastAPI instance.
app = FastAPI(
    title="Preference Extraction API",
    description="Extracts customer preference scores from an input string using a PEARL-inspired architecture.",
    version="1.0"
)

@app.post("/extract", response_model=PreferenceOutput)
def extract_preferences(payload: PreferenceInput):
    if not payload.input_text.strip():
        raise HTTPException(status_code=400, detail="Input text is empty or invalid.")
    
    # Use the preference extractor function.
    prefs = pearl_extract_preferences_single(payload.input_text, exemplar_weight=payload.exemplar_weight)
    return PreferenceOutput(preferences=prefs)

# Run the app when invoked directly.
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)