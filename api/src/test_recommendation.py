import pandas as pd
import psycopg2
from difflib import get_close_matches

# --- Connect to PostgreSQL and Load Data ---
print("\n🔹 Loading Data from PostgreSQL...")

conn = psycopg2.connect(
    dbname="travel_db",
    user="postgres",
    password="alexa",  # Replace with your actual password
    host="localhost",
    port="5432"
)

# Load user preferences and places data
user_df = pd.read_sql("SELECT * FROM user_preferences;", conn)
places_df = pd.read_sql("SELECT * FROM places;", conn)
conn.close()

# --- Debugging: Print Data Preview ---
print("\n🔹 User Preferences Data (First 5 rows):")
print(user_df.head())

print("\n🔹 Places Data (First 5 rows):")
print(places_df.head())

# --- Check Available Categories ---
print("\n🔹 Sample Categories from Places Data (After Fix):")
print(places_df["category"].value_counts())  # Show unique categories and counts

print("\n🔹 User Preferences Categories (Columns in user_preferences):")
user_prefs_keys = list(user_df.columns[2:])  # Exclude user_id and username
print(user_prefs_keys)

# --- Find a Valid User ID ---
valid_user_ids = user_df["user_id"].unique()
if len(valid_user_ids) == 0:
    print("\n⚠️ ERROR: No users found in `user_preferences`. Please check the database!")
    exit()

test_user_id = valid_user_ids[0]  # Use the first available user_id
print(f"\n🔹 Using test user_id: {test_user_id}")



# --- Recommendation Function: Improved Matching ---
def compute_score(row):
    """Calculate the score of a place based on user preferences."""
    place_cat = str(row["category"]).lower() if row["category"] and row["category"] != "unknown" else ""
    score = 0

    for cat in user_prefs_keys:
        keyword = cat.replace("_", " ").lower()  # Convert category to lowercase
        if keyword in place_cat or get_close_matches(keyword, [place_cat], cutoff=0.6):
            user_rating = user_df.loc[user_df["user_id"] == test_user_id, cat].values[0]
            print(f"🔹 Debug: Matching '{keyword}' -> Place Category: '{place_cat}' | User Score: {user_rating}")  # Debugging

            score = max(score, user_rating)  # Keep the highest rating match

    return score

# --- Apply the Scoring Function ---
places_df["score"] = places_df.apply(compute_score, axis=1)

# --- Sort by Score & Get Top Recommendations ---
recommendations = places_df.sort_values("score", ascending=False).head(5)

# --- Display Recommended Places ---
print("\n🔹 **Recommended Places:**")
print(recommendations[["name", "category", "address", "score"]])
