# Core Python imports
import os
import math

# Data processing and analysis
import numpy as np
import pandas as pd
import joblib

# Deep learning
from tensorflow.keras.models import load_model

# Web framework and visualization
import streamlit as st
import pydeck as pdk

# Network and routing
import networkx as nx
from openrouteservice import convert

# HTTP requests
import requests  # For calling the API

# Import recommender models from new structure
from src.recommenders import (
    AutoencoderRecommender, 
    SVDPlaceRecommender, 
    TransferRecommender,
    EnsembleRecommender
)

# Path optimization
from pathway import get_optimal_path
from pathway import reorder_with_tsp

# Initialize session state for user authentication
if 'user_token' not in st.session_state:
    st.session_state.user_token = None
if 'username' not in st.session_state:
    st.session_state.username = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'saved_preferences' not in st.session_state:
    st.session_state.saved_preferences = {}
if 'saved_reviews' not in st.session_state:
    st.session_state.saved_reviews = {}
if 'current_recommendations' not in st.session_state:
    st.session_state.current_recommendations = None
if 'current_method' not in st.session_state:
    st.session_state.current_method = None

def login_user(username: str, password: str):
    try:
        # First check if user exists
        check_user = requests.get(
            f"http://localhost:8000/users/{username}/exists",
            headers={"Content-Type": "application/json"}
        )
        if check_user.status_code == 404:
            st.error("Username doesn't exist. Please sign up first.")
            return False

        # Try to login
        response = requests.post(
            "http://localhost:8000/token",
            data={"username": username, "password": password}
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state.user_token = data["access_token"]
            st.session_state.username = username
            return True
        elif response.status_code == 401:
            st.error("Incorrect password. Please try again.")
            return False
        else:
            st.error("Login failed. Please try again.")
            return False
    except Exception as e:
        st.error(f"Login failed: {str(e)}")
        return False

def register_user(username: str, email: str, password: str):
    try:
        response = requests.post(
            "http://localhost:8000/users/",
            json={"username": username, "email": email, "password": password}
        )
        if response.status_code == 200:
            st.success("Registration successful! Please log in.")
            return True
        else:
            st.error(f"Registration failed: {response.json()['detail']}")
            return False
    except Exception as e:
        st.error(f"Registration failed: {str(e)}")
        return False

# Show login/signup if not authenticated
if not st.session_state.user_token:
    st.title("Welcome to Place Recommender")
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        st.header("Login")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            if login_user(login_username, login_password):
                st.success("Logged in successfully!")
                st.rerun()
    
    with tab2:
        st.header("Sign Up")
        reg_username = st.text_input("Username", key="reg_username")
        reg_email = st.text_input("Email", key="reg_email")
        reg_password = st.text_input("Password", type="password", key="reg_password")
        if st.button("Register"):
            if register_user(reg_username, reg_email, reg_password):
                st.info("Please proceed to login.")
    st.stop()

# Add logout button in sidebar if logged in
if st.sidebar.button("Logout"):
    st.session_state.user_token = None
    st.session_state.username = None
    st.rerun()

st.sidebar.write(f"Logged in as: {st.session_state.username}")

# After login, load saved preferences and reviews
if st.session_state.user_token:
    try:
        # Load preferences
        pref_response = requests.get(
            "http://localhost:8000/preferences/",
            headers={"Authorization": f"Bearer {st.session_state.user_token}"}
        )
        if pref_response.status_code == 200:
            st.session_state.saved_preferences = {
                pref['category']: pref['rating'] 
                for pref in pref_response.json()
            }
            # Set the sliders for saved preferences
            for cat, rating in st.session_state.saved_preferences.items():
                st.session_state[f"slider_{cat}"] = rating
                st.session_state[f"chk_{cat}"] = True
        
        # Load reviews
        review_response = requests.get(
            "http://localhost:8000/reviews/",
            headers={"Authorization": f"Bearer {st.session_state.user_token}"}
        )
        if review_response.status_code == 200:
            reviews = review_response.json()
            # Store reviews by place_id
            st.session_state.saved_reviews = {
                review['place_id']: {
                    'rating': review['rating'],
                    'comment': review['comment'],
                    'submitted': True,
                    'previous_submission': True
                }
                for review in reviews
            }
    except Exception as e:
        st.error(f"Error loading user data: {str(e)}")

# ---------------------------
# Helper Functions
# ---------------------------
def haversine(lat1, lon1, lat2, lon2):
    """Compute distance (in meters) between two lat/lon points."""
    R = 6371000  # Earth's radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def process_types(types_str):
    """Convert a comma‑separated string of place types to a list of lowercase strings."""
    if pd.isna(types_str):
        return []
    return [t.strip().lower() for t in types_str.split(',')]

def euclidean_distance(p1, p2):
    """Euclidean distance between two points (lat,lon)."""
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def show_map(recs, ors_key, profile="foot-walking"):
    """Display an interactive PyDeck map with an optimized route and text labels."""
    if not recs:
        return

    import openrouteservice
    from openrouteservice import exceptions

    if 'row' in recs[0]:
        names = [cand['row']['name'] for cand in recs]
        lats = [cand['row']['lat'] for cand in recs]
        lons = [cand['row']['lng'] for cand in recs]
    else:
        names = [cand['name'] for cand in recs]
        lats = [cand['lat'] for cand in recs]
        lons = [cand['lng'] for cand in recs]

    coords = list(zip(lons, lats))  # (lng, lat)
    map_df = pd.DataFrame({
        'name': names,
        'lat': lats,
        'lon': lons
    })
    route_coords = None
    if not map_df.empty:
        if ors_key and ors_key != "":
            try:
                client = openrouteservice.Client(key=ors_key)
                coords = map_df[['lon', 'lat']].values.tolist()
                route_geojson = client.directions(coords, profile=profile, format='geojson')
                route_coords = route_geojson['features'][0]['geometry']['coordinates']
            except Exception as e:
                st.error(f"Routing API error: {e}")
                route_coords = None
        if route_coords is None:
            G = nx.complete_graph(len(map_df))
            coords = map_df[['lat', 'lon']].values
            for i, j in G.edges():
                G[i][j]['weight'] = euclidean_distance((coords[i][0], coords[i][1]),
                                                        (coords[j][0], coords[j][1]))
            tsp_route = nx.approximation.traveling_salesman_problem(G, weight='weight')
            optimized_data = map_df.iloc[tsp_route].reset_index(drop=True)
            route_coords = optimized_data[['lon', 'lat']].values.tolist()
        else:
            optimized_data = map_df.copy()
        
        text_layer = pdk.Layer(
            "TextLayer",
            data=optimized_data,
            get_position='[lon, lat]',
            get_text='name',
            get_color='[255, 255, 255, 255]',
            get_size=16,
            get_angle=0,
            anchor='middle'
        )
        path_layer = pdk.Layer(
            "PathLayer",
            data=[{"path": route_coords}],
            get_path="path",
            get_color="[0, 0, 255]",
            width_scale=20,
            width_min_pixels=3,
        )
        view_state = pdk.ViewState(
            latitude=optimized_data['lat'].mean(),
            longitude=optimized_data['lon'].mean(),
            zoom=14,
            pitch=0,
        )
        deck = pdk.Deck(
            layers=[path_layer, text_layer],
            initial_view_state=view_state,
            tooltip={"text": "{name}"}
        )
        st.pydeck_chart(deck)

def display_recommendation(rec):
    with st.container():
        cols = st.columns([1, 3])
        with cols[0]:
            st.image(rec.get('icon', 'https://via.placeholder.com/80'), width=80)
        with cols[1]:
            st.markdown(f"**Name:** {rec.get('name', 'N/A')}")
            st.markdown(f"**Average Rating:** {rec.get('actual_rating', rec.get('rating', 'N/A'))}")
            st.markdown(f"**User Ratings Total:** {rec.get('user_ratings_total', 'N/A')}")
            st.markdown(f"**Distance:** {rec.get('distance', 0):.2f} m")
            st.markdown(f"**Score:** {rec.get('score', 0):.2f}")
            with st.expander("Show More Details"):
                st.write(f"**Types:** {rec.get('types', 'N/A')}")
                st.write(f"**Description:** {rec.get('description', 'No description available.')}")
                st.write(f"**Coordinates:** (Lat: {rec.get('lat', 'N/A')}, Lon: {rec.get('lng', 'N/A')})")

def optimize_and_display_route(recommendations, user_lat, user_lng, ors_key, profile):
    recs_for_map = [{'row': rec} for rec in recommendations] if 'row' not in recommendations[0] else recommendations

    # Always apply RL optimization
    st.subheader("Optimized Route Based on Preferences + Distance")
    valid_recs = [
        p if 'row' not in p else p['row'] for p in recs_for_map
        if (p.get("lat") if 'row' not in p else p['row'].get("lat")) is not None and
           (p.get("lng") if 'row' not in p else p['row'].get("lng")) is not None
    ]
    try:
        rl_path = get_optimal_path(valid_recs, user_lat, user_lng)
        if rl_path:
            # Reorder RL output using TSP (Haversine)
            rl_path = reorder_with_tsp(rl_path)
            recs_for_map = [{'row': rec} for rec in rl_path]
            st.success("RL + TSP Reordering Applied Successfully!")
    except Exception as e:
        st.warning(f"RL optimization failed: {e}")

    st.subheader("Map View: Recommended Places & Optimized Route")
    st.markdown(f"**Routing Mode:** `{profile.replace('-', ' ').capitalize()}`")
    show_map(recs_for_map, ors_key, profile=profile)

    # Route Breakdown
    st.subheader("Route Details")
    total_distance = 0
    for i in range(len(recs_for_map)):
        current = recs_for_map[i]['row'] if 'row' in recs_for_map[i] else recs_for_map[i]
        st.markdown(f"**{i+1}. {current['name']}** *(Category: {current.get('category', 'N/A')})*")

        if i == 0:
            dist = haversine(user_lat, user_lng, current['lat'], current['lng']) / 1000
            total_distance += dist
            st.write(f"\u21b3 Distance from start location: `{dist:.2f} km`")
        else:
            prev = recs_for_map[i - 1]['row'] if 'row' in recs_for_map[i - 1] else recs_for_map[i - 1]
            dist = haversine(prev['lat'], prev['lng'], current['lat'], current['lng']) / 1000
            total_distance += dist
            st.write(f"\u21b3 Distance from previous: `{dist:.2f} km`")
    st.markdown(f"**Total Travel Distance:** `{total_distance:.2f} km`")

# ---------------------------
# Load Resources
# ---------------------------
# Update load_resources to include type processing

BASE_PATH = "reco/streamlit/" # Don't edit this path, streamlit app will break

@st.cache(allow_output_mutation=True)
def load_resources():
    if os.environ.get('ENV') == 'prod':
        auto_model = load_model(os.path.join(BASE_PATH, "models/autoencoder.h5"))
        scaler = joblib.load(os.path.join(BASE_PATH, "models/scaler.save"))
        places = pd.read_csv(os.path.join(BASE_PATH, "resources/combined_places.csv"))
    else:
        auto_model = load_model("models/autoencoder.h5")
        scaler = joblib.load("models/scaler.save")
        places = pd.read_csv("resources/combined_places.csv")
    places['types_processed'] = places['types'].apply(process_types)  # Add this line
    return auto_model, scaler, places

auto_model, scaler, places = load_resources()

# ---------------------------
# Categories
# ---------------------------
categories = [
    "resorts", "burger/pizza shops", "hotels/other lodgings", "juice bars", "beauty & spas",
    "gardens", "amusement parks", "farmer market", "market", "music halls", "nature",
    "tourist attractions", "beaches", "parks", "theatres", "museums", "malls", "restaurants",
    "pubs/bars", "local services", "art galleries", "dance clubs", "swimming pools", "bakeries",
    "cafes", "view points", "monuments", "zoo", "supermarket"
]

category_to_place_types = {
    "resorts": ["lodging"],
    "burger/pizza shops": ["restaurant"],
    "hotels/other lodgings": ["lodging"],
    "juice bars": ["restaurant", "cafe"],
    "beauty & spas": ["beauty_salon", "spa"],
    "gardens": ["park", "tourist_attraction"],
    "amusement parks": ["amusement_park"],
    "farmer market": ["grocery_or_supermarket"],
    "market": ["grocery_or_supermarket"],
    "music halls": ["establishment"],
    "nature": ["park", "tourist_attraction"],
    "tourist attractions": ["tourist_attraction"],
    "beaches": ["tourist_attraction"],
    "parks": ["park"],
    "theatres": ["movie_theater"],
    "museums": ["museum"],
    "malls": ["shopping_mall"],
    "restaurants": ["restaurant"],
    "pubs/bars": ["bar"],
    "local services": ["establishment"],
    "art galleries": ["art_gallery"],
    "dance clubs": ["night_club"],
    "swimming pools": ["establishment"],
    "bakeries": ["bakery"],
    "cafes": ["cafe"],
    "view points": ["tourist_attraction"],
    "monuments": ["tourist_attraction"],
    "zoo": ["zoo"],
    "supermarket": ["supermarket", "grocery_or_supermarket"]
}

# Initialize recommenders - create instances here for later use 
# (instead of defining classes in app.py)
autoencoder_recommender = AutoencoderRecommender(
    auto_model=auto_model,
    scaler=scaler,
    places_df=places,
    categories_list=categories,
    category_mappings=category_to_place_types
)

svd_recommender = SVDPlaceRecommender(
    category_to_place_types=category_to_place_types
)
svd_recommender.fit(places)

transfer_recommender = TransferRecommender()
transfer_recommender.train_base_model()
transfer_recommender.transfer_to_places(places)

# Initialize the ensemble recommender
ensemble_recommender = EnsembleRecommender()
ensemble_recommender.initialize_models(
    auto_model=auto_model,
    scaler=scaler,
    places_df=places,
    category_to_place_types=category_to_place_types
)

# ---------------------------
# Chat & UI Setup
# ---------------------------
st.sidebar.header("User Settings")
user_lat = st.sidebar.number_input("Latitude", value=40.4168, format="%.4f")
user_lng = st.sidebar.number_input("Longitude", value=-3.7038, format="%.4f")
ors_key = st.sidebar.text_input("OpenRouteService API Key (optional)", value="", type="password")
method = st.sidebar.selectbox("Method", ["Autoencoder-Based", "SVD-Based", "Transfer-Based", "Ensemble"])
profile = st.sidebar.selectbox(
    "Routing Profile",
    options=["foot-walking", "driving-car"],
    index=0,
    help="Choose between walking or driving for route optimization."
)

# Add ensemble weights if ensemble method is selected
if method == "Ensemble":
    st.sidebar.subheader("Ensemble Weights")
    auto_weight = st.sidebar.slider("Autoencoder Weight", 0.0, 1.0, 0.33, 0.01)
    svd_weight = st.sidebar.slider("SVD Weight", 0.0, 1.0, 0.33, 0.01)
    transfer_weight = st.sidebar.slider("Transfer Learning Weight", 0.0, 1.0, 0.34, 0.01)
    # Normalize weights to ensure they sum to 1
    total_weight = auto_weight + svd_weight + transfer_weight
    if total_weight > 0:
        auto_weight = auto_weight / total_weight
        svd_weight = svd_weight / total_weight
        transfer_weight = transfer_weight / total_weight
    ensemble_weights = {
        'autoencoder': auto_weight,
        'svd': svd_weight,
        'transfer': transfer_weight
    }

num_recs = 5

st.title("Personalized Place Recommendations in Madrid")
st.write("Rate your preferences. Check a category and use the slider (0–5) for those you care about.")
st.subheader("💬 Chat with the Recommender")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_msg = st.chat_input("Tell me what you're looking for (e.g., 'I love nature and museums')")

if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    # Call the preference extraction API with authentication
    headers = {"Authorization": f"Bearer {st.session_state.user_token}"}
    try:
        response = requests.post(
            "http://localhost:8000/extract-preferences",
            json={"text": user_msg},
            headers=headers
        )
        if response.status_code == 200:
            extracted_prefs = response.json()["preferences"]
            if not extracted_prefs:
                bot_reply = "Hmm, I couldn't find any familiar categories. Try mentioning something like 'parks, cafes, museums'."
            else:
                # Update saved preferences with new ones
                st.session_state.saved_preferences.update(extracted_prefs)
                
                # Update the UI sliders and save to database
                for cat, rating in extracted_prefs.items():
                    st.session_state[f"slider_{cat}"] = rating
                    st.session_state[f"chk_{cat}"] = True
                    try:
                        requests.post(
                            "http://localhost:8000/preferences/",
                            json={"category": cat, "rating": rating},
                            headers=headers
                        )
                    except Exception as e:
                        st.error(f"Error saving preference for {cat}: {str(e)}")
                
                bot_reply = "Great! I've updated your preferences. You can adjust them below if needed."
        else:
            bot_reply = "Sorry, I couldn't process your preferences right now."
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        st.rerun()
    except Exception as e:
        st.error(f"Error processing preferences: {str(e)}")

# Modify get_user_inputs to use saved preferences
def get_user_inputs():
    input_ratings = []
    provided_mask = []
    
    # Create columns for better layout
    cols = st.columns([1, 2])
    with cols[0]:
        st.write("**Select Categories**")
    with cols[1]:
        st.write("**Rate (0-5)**")
    
    for cat in categories:
        default_chk = st.session_state.get(f"chk_{cat}", False)
        default_val = st.session_state.get(f"slider_{cat}", 
                     st.session_state.saved_preferences.get(cat, 0.0))
        
        cols = st.columns([1, 2])
        with cols[0]:
            provide = st.checkbox(f"{cat}", value=default_chk, key=f"chk_{cat}")
        with cols[1]:
            if provide:
                rating = st.slider("", 
                                min_value=0.0, max_value=5.0, 
                                step=0.5, value=default_val, 
                                key=f"slider_{cat}")
                
                # Save preference to database if it's changed
                if rating != st.session_state.saved_preferences.get(cat, 0.0):
                    try:
                        response = requests.post(
                            "http://localhost:8000/preferences/",
                            json={"category": cat, "rating": rating},
                            headers={"Authorization": f"Bearer {st.session_state.user_token}"}
                        )
                        if response.status_code == 200:
                            st.session_state.saved_preferences[cat] = rating
                    except Exception as e:
                        st.error(f"Error saving preference for {cat}: {str(e)}")
                
                input_ratings.append(rating)
                provided_mask.append(True)
            else:
                # If checkbox is unchecked, remove preference from database
                if cat in st.session_state.saved_preferences:
                    try:
                        response = requests.delete(
                            f"http://localhost:8000/preferences/{cat}",
                            headers={"Authorization": f"Bearer {st.session_state.user_token}"}
                        )
                        if response.status_code == 200:
                            st.session_state.saved_preferences.pop(cat, None)
                    except Exception as e:
                        st.error(f"Error removing preference for {cat}: {str(e)}")
                input_ratings.append(0.0)
                provided_mask.append(False)
    
    return np.array(input_ratings, dtype=np.float32).reshape(1, -1), np.array(provided_mask, dtype=bool).reshape(1, -1)

input_ratings, provided_mask = get_user_inputs()
user_preferences_dict = {cat: float(st.session_state.get(f"slider_{cat}", 0.0)) for cat in categories}

# Modify the Generate Recommendations button section
if st.button("Generate Recommendations"):
    # Store the current method
    st.session_state.current_method = method
    
    # Compute predicted ratings using the autoencoder (for both methods)
    input_scaled = scaler.transform(input_ratings)
    predicted_scaled = auto_model.predict(input_scaled)
    predicted_scaled = np.clip(predicted_scaled, 0, 1)
    predicted = scaler.inverse_transform(predicted_scaled)
    final_predictions = predicted[0]
    final_predictions[provided_mask[0]] = input_ratings[0][provided_mask[0]]
    predicted_ratings_dict = {cat: final_predictions[i] for i, cat in enumerate(categories)}


    with st.expander("Show Predicted Category Ratings"):
        for cat, rating in predicted_ratings_dict.items():
            st.write(f"**{cat}:** {rating:.2f}")

    # Depending on the method, call the corresponding recommender
    if method == "Autoencoder-Based":
        st.subheader("Autoencoder-Based Recommendations")
        
        # Convert dictionary of preferences to array format and mask
        user_prefs_array = np.array([predicted_ratings_dict.get(cat, 0) for cat in categories])
        provided_mask = np.array([cat in user_preferences_dict and user_preferences_dict[cat] > 0 
                                  for cat in categories])
        
        recommendations = autoencoder_recommender.get_recommendations(
            user_lat=user_lat,
            user_lon=user_lng,
            user_prefs=user_prefs_array,
            provided_mask=provided_mask,
            num_recs=num_recs
        )
        
        # Store recommendations in session state
        st.session_state.current_recommendations = recommendations if recommendations else None
        
    elif method == "SVD-Based":
        st.subheader("SVD-Based Recommendations")
        with st.expander("SVD Model Evaluation Details"):
            eval_metrics = svd_recommender.evaluate_model(places)
            st.write(f"RMSE: {eval_metrics['rmse_mean']:.4f} (+/- {eval_metrics['rmse_std']:.4f})")
            st.write(f"MAE:  {eval_metrics['mae_mean']:.4f} (+/- {eval_metrics['mae_std']:.4f})")
            
        recommendations = svd_recommender.get_recommendations(
            df=places, 
            user_lat=user_lat, 
            user_lon=user_lng, 
            predicted_ratings=predicted_ratings_dict, 
            top_n=num_recs, 
            max_distance=5
        )
        # Store recommendations in session state
        st.session_state.current_recommendations = recommendations
        
    elif method == "Transfer-Based":
        st.subheader("Transfer-Based Recommendations")
        recommendations = transfer_recommender.get_recommendations(
            user_preferences=user_preferences_dict,
            user_lat=user_lat,
            user_lon=user_lng,
            places_df=places,
            top_n=num_recs
        )
        # Store recommendations in session state
        st.session_state.current_recommendations = recommendations
        
    elif method == "Ensemble":
        st.subheader("Ensemble-Based Recommendations")
        # Update weights if they were customized
        custom_weights = {
            'autoencoder': auto_weight,
            'svd': svd_weight,
            'transfer': transfer_weight
        }
        ensemble_recommender.weights = custom_weights
        
        recommendations = ensemble_recommender.get_recommendations(
            user_lat=user_lat,
            user_lon=user_lng,
            user_prefs=user_preferences_dict,
            predicted_ratings_dict=predicted_ratings_dict,
            num_recs=num_recs
        )
        # Store recommendations in session state
        st.session_state.current_recommendations = recommendations

# After the Generate Recommendations button section, add this code to display recommendations
if st.session_state.current_recommendations:
    method = st.session_state.current_method
    recommendations = st.session_state.current_recommendations
    
    st.subheader(f"Top Place Recommendations ({method})")
    if recommendations:
        st.markdown("### Best Recommendation")
        display_recommendation(recommendations[0])
        st.markdown("### Other Recommendations")
        for rec in recommendations[1:]:
            display_recommendation(rec)
        
        # Show map
        st.subheader("Map View: Recommended Places & Optimized Route")
        optimize_and_display_route(recommendations, user_lat, user_lng, ors_key, profile)
    else:
        st.error(f"No recommendations found with {method} method.")
