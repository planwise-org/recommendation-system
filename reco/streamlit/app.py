import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import math
import pydeck as pdk
import networkx as nx
import openrouteservice
from openrouteservice import convert
import re
import requests  # For calling the API
import textblob
import spacy
import json
from datetime import datetime
import os

# For SVD-based recommendations:
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
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
            f"http://localhost:8080/api/users/{username}/exists",
            headers={"Content-Type": "application/json"}
        )
        if check_user.status_code == 404:
            st.error("Username doesn't exist. Please sign up first.")
            return False

        # Try to login
        response = requests.post(
            "http://localhost:8080/api/token",
            data={"username": username, "password": password}
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state.user_token = data["access_token"]
            st.session_state.username = username
            return True
        elif response.status_code == 401:
            error_detail = response.json().get('detail', 'Incorrect password')
            st.error(f"Login failed: {error_detail}")
            return False
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            st.error(f"Login failed: {error_detail}")
            return False
    except Exception as e:
        st.error(f"Login failed: {str(e)}")
        return False

def register_user(username: str, password: str):
    try:
        response = requests.post(
            "http://localhost:8080/api/users/",
            json={
                "username": username,
                "password": password,
                "full_name": username,  # Using username as full_name for now
                "role": "user"  # This will be converted to UserRole.USER by the API
            }
        )
        if response.status_code in [200, 201]:  # Accept both 200 and 201
            st.success("Registration successful! Please log in.")
            return True
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            st.error(f"Registration failed: {error_detail}")
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
        reg_password = st.text_input("Password", type="password", key="reg_password")
        if st.button("Register"):
            if register_user(reg_username, reg_password):
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
            "http://localhost:8080/api/preferences/",
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
            "http://localhost:8080/api/reviews/",
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
            
            # Add review section
            place_id = rec.get('place_id')
            if place_id:
                # Check if user has already reviewed this place
                try:
                    review_response = requests.get(
                        f"http://localhost:8080/api/reviews/user/{place_id}",
                        headers={"Authorization": f"Bearer {st.session_state.user_token}"}
                    )
                    if review_response.status_code == 200:
                        review_data = review_response.json()
                        if review_data['submitted']:
                            st.markdown("**Your Review:**")
                            st.markdown(f"Rating: {review_data['rating']} ⭐")
                            if review_data['comment']:
                                st.markdown(f"Comment: {review_data['comment']}")
                        else:
                            with st.form(f"review_form_{place_id}"):
                                st.markdown("**Write a Review**")
                                rating = st.slider("Rating", 1.0, 5.0, 3.0, 0.5)
                                comment = st.text_area("Comment (optional)")
                                if st.form_submit_button("Submit Review"):
                                    try:
                                        response = requests.post(
                                            "http://localhost:8080/api/reviews/",
                                            json={
                                                "place_id": place_id,
                                                "rating": rating,
                                                "comment": comment
                                            },
                                            headers={"Authorization": f"Bearer {st.session_state.user_token}"}
                                        )
                                        if response.status_code in [200, 201]:
                                            st.success("Review submitted successfully!")
                                            st.rerun()
                                        else:
                                            st.error(f"Failed to submit review. Status code: {response.status_code}")
                                            if response.text:
                                                st.error(f"Error details: {response.text}")
                                    except Exception as e:
                                        st.error(f"Error submitting review: {str(e)}")
                except Exception as e:
                    st.error(f"Error loading review data: {str(e)}")

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


class BaseRecommender:
    def get_recommendations(self, user_lat, user_lon, user_prefs, num_recs):
        raise NotImplementedError("Subclasses must implement this method.")

# ---------------------------
# Autoencoder-Based Recommendation
# ---------------------------
class AutoencoderRecommender(BaseRecommender):
    def get_recommendations(self, user_lat, user_lon, user_prefs, provided_mask, num_recs):
        # Reconstruct predicted ratings
        input_scaled = scaler.transform([user_prefs])
        predicted_scaled = auto_model.predict(input_scaled)
        predicted = scaler.inverse_transform(np.clip(predicted_scaled, 0, 1))[0]
        final_predictions = np.where(provided_mask, user_prefs, predicted)

        # Candidate scoring logic
        candidates = []
        for _, row in places.iterrows():
            dist = haversine(user_lat, user_lon, row['lat'], row['lng'])
            if dist > 2000:
                continue

            best_cat = None
            best_score = 0
            for i, cat in enumerate(categories):
                mapped_types = category_to_place_types.get(cat, [])
                if any(pt in row['types_processed'] for pt in mapped_types):
                    cat_score = final_predictions[i] / 5.0
                    if cat_score > best_score:
                        best_score = cat_score
                        best_cat = cat

            if best_cat:
                norm_rating = (row['rating'] / 5.0) if pd.notna(row['rating']) else 0
                norm_reviews = (np.log(row['user_ratings_total'] + 1)) / np.log(places['user_ratings_total'].max() + 1) if pd.notna(row['user_ratings_total']) else 0
                score = (0.1 * (1 - dist/2000) +
                        0.2 * norm_rating +
                        0.2 * norm_reviews +
                        0.5 * best_score)

                candidates.append({
                    'row': row,
                    'score': score,
                    'category': best_cat,
                    'distance': dist
                })

        # Category balancing logic
        candidates.sort(key=lambda x: x['score'], reverse=True)
        category_groups = {}
        for candidate in candidates:
            category_groups.setdefault(candidate['category'], []).append(candidate)

        final_recs = []
        round_idx = 0
        while len(final_recs) < num_recs:
            added = False
            for cat in category_groups.values():
                if len(cat) > round_idx:
                    final_recs.append(cat[round_idx])
                    added = True
                    if len(final_recs) >= num_recs:
                        break
            if not added:
                break
            round_idx += 1

        return [{
            'name': r['row']['name'],
            'lat': r['row']['lat'],
            'lng': r['row']['lng'],
            'score': r['score'],
            'category': r['category'],
            'rating': r['row']['rating'],
            'reviews': r['row']['user_ratings_total']
        } for r in final_recs[:num_recs]]


# ---------------------------
# SVDPlaceRecommender: SVD-Based Collaborative Filtering
# ---------------------------
class SVDPlaceRecommender:
    def __init__(self, svd_params=None):
        self.svd_params = svd_params or {
            "n_factors": 150,
            "n_epochs": 10,
            "lr_all": 0.002,
            "reg_all": 0.02,
            "random_state": 42,
            "verbose": False
        }
        self.model = SVD(**self.svd_params)
        self.category_to_place_types = category_to_place_types

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        r = 6371
        return c * r

    def get_type_score(self, place_types, predicted_ratings):
        if pd.isna(place_types):
            return 0
        types_list = str(place_types).split(', ')
        scores = []
        type_preferences = {}
        for category, rating in predicted_ratings.items():
            if category in self.category_to_place_types:
                for pt in self.category_to_place_types[category]:
                    type_preferences[pt] = max(type_preferences.get(pt, 0), rating)
        for pt in types_list:
            if pt in type_preferences:
                scores.append(type_preferences[pt])
        return np.mean(scores) if scores else 0

    def prepare_data(self, df):
        reader = Reader(rating_scale=(1, 5))
        ratings_dict = {'user': [], 'item': [], 'rating': []}
        for idx, row in df.iterrows():
            ratings_dict['item'].append(row['place_id'])
            ratings_dict['user'].append(f'user_{idx % 100}')
            ratings_dict['rating'].append(row['rating'] if pd.notna(row['rating']) else 3.0)
        ratings_df = pd.DataFrame(ratings_dict)
        return Dataset.load_from_df(ratings_df[['user', 'item', 'rating']], reader)

    def evaluate_model(self, df):
        data = self.prepare_data(df)
        cv_results = cross_validate(self.model, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
        with st.expander("SVD Model Evaluation Details"):
            st.write(f"RMSE: {cv_results['test_rmse'].mean():.4f} (+/- {cv_results['test_rmse'].std():.4f})")
            st.write(f"MAE:  {cv_results['test_mae'].mean():.4f} (+/- {cv_results['test_mae'].std():.4f})")
        return cv_results

    def fit(self, df):
        data = self.prepare_data(df)
        trainset = data.build_full_trainset()
        self.model.fit(trainset)

    def get_recommendations(self, df, user_lat, user_lon, predicted_ratings, top_n=10, max_distance=5):
        predictions = []
        for idx, row in df.iterrows():
            if pd.isna(row['rating']):
                continue
            distance = self.haversine_distance(user_lat, user_lon, row['lat'], row['lng'])
            if distance > max_distance:
                continue
            svd_pred = self.model.predict('user_0', row['place_id']).est
            type_score = self.get_type_score(row['types'], predicted_ratings)
            distance_score = 1 / (1 + distance * 0.1)
            score = (0.4 * svd_pred/5.0 +
                     0.4 * type_score/5.0 +
                     0.2 * distance_score)
            predictions.append({
                'place_id': row['place_id'],
                'name': row['name'],
                'predicted_rating': svd_pred,
                'type_score': type_score,
                'actual_rating': row['rating'],
                'user_ratings_total': row['user_ratings_total'],
                'distance': distance,
                'types': row['types'],
                'vicinity': row['vicinity'],
                'icon': row['icon'],
                'description': row.get('description', ''),
                'score': score,
                'lat': row['lat'],
                'lng': row['lng']
            })

        return sorted(predictions, key=lambda x: x['score'], reverse=True)[:top_n]


# ---------------------------
# Transfer-Based Recommendation Wrapper
# ---------------------------
from src.transfer_recommender import TransferRecommender

class TransferBasedRecommender:
    def __init__(self):
        self.tr = TransferRecommender()


        if os.environ.get('ENV') == 'prod':
            self.tr.train_base_model(save_path= BASE_PATH + "models")
            self.places = pd.read_csv(os.path.join(BASE_PATH, "resources/combined_places.csv"))
            self.tr.transfer_to_places(self.places, save_path= BASE_PATH+"models")

        else:
            self.tr.train_base_model(save_path="models")
            self.places = pd.read_csv("resources/combined_places.csv")
            self.tr.transfer_to_places(self.places, save_path="models")

        self.places['types_processed'] = self.places['types'].apply(process_types)


    def get_recommendations(self, user_lat, user_lon, user_prefs, num_recs):
        return self.tr.get_recommendations(
            user_preferences=user_prefs,
            user_lat=user_lat,
            user_lon=user_lon,
            places_df=self.places,
            top_n=num_recs
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
            "http://localhost:8080/api/preferences/extract-preferences",
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
                        response = requests.post(
                            "http://localhost:8080/api/preferences/",
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
                            "http://localhost:8080/api/preferences/",
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
                            f"http://localhost:8080/api/preferences/{cat}",
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
        candidates = []
        for idx, row in places.iterrows():
            dist = haversine(user_lat, user_lng, row['lat'], row['lng'])
            if dist > 2000:
                continue
            best_cat = None
            best_factor = 0
            for cat, pred_val in predicted_ratings_dict.items():
                mapped_types = category_to_place_types.get(cat, [])
                if any(pt in row['types_processed'] for pt in mapped_types):
                    normalized = pred_val / 5.0
                    if normalized > best_factor:
                        best_factor = normalized
                        best_cat = cat
            if best_cat is None:
                continue
            candidates.append({
                'row': row.to_dict(),
                'distance': dist,
                'pred_factor': best_factor,
                'category': best_cat
            })
        
        if not candidates:
            st.error("No candidate places found within 2 km.")
            st.session_state.current_recommendations = None
        else:
            max_reviews = max(
                [cand['row'].get('user_ratings_total', 1) for cand in candidates
                if pd.notna(cand['row'].get('user_ratings_total'))] or [1]
            )
            w_distance = 0.1
            w_rating = 0.2
            w_reviews = 0.2
            w_pred = 0.5
            
            for candidate in candidates:
                row_dict = candidate['row']
                dist = candidate['distance']
                distance_score = 1 - (dist / 2000)
                norm_rating = (row_dict.get('rating', 0) / 5.0) if row_dict.get('rating') is not None else 0
                norm_reviews = (np.log(row_dict.get('user_ratings_total', 0) + 1) / np.log(max_reviews + 1)) if row_dict.get('user_ratings_total') is not None else 0
                candidate['score'] = (w_distance * distance_score +
                                    w_rating * norm_rating +
                                    w_reviews * norm_reviews +
                                    w_pred * candidate['pred_factor'])
            
            groups = {}
            for candidate in candidates:
                groups.setdefault(candidate['category'], []).append(candidate)
            for cat in groups:
                groups[cat] = sorted(groups[cat], key=lambda x: x['score'], reverse=True)
            
            final_candidates = []
            round_idx = 0
            while len(final_candidates) < 5:
                added_this_round = False
                for cat, cand_list in groups.items():
                    if len(cand_list) > round_idx:
                        final_candidates.append(cand_list[round_idx])
                        added_this_round = True
                        if len(final_candidates) == 5:
                            break
                if not added_this_round:
                    break
                round_idx += 1

            final_candidates = sorted(final_candidates, key=lambda x: x['score'], reverse=True)
            
            flat_final_candidates = []
            for candidate in final_candidates:
                row = candidate['row']
                flat_candidate = {
                    'name': row.get('name', 'N/A'),
                    'actual_rating': row.get('rating', row.get('actual_rating', 'N/A')),
                    'user_ratings_total': row.get('user_ratings_total', 'N/A'),
                    'distance': candidate.get('distance', 0),
                    'types': row.get('types', 'N/A'),
                    'vicinity': row.get('vicinity', 'N/A'),
                    'icon': row.get('icon', 'https://via.placeholder.com/80'),
                    'description': row.get('description', 'No description available.'),
                    'score': candidate.get('score', 0),
                    'lat': row.get('lat', 'N/A'),
                    'lng': row.get('lng', 'N/A'),
                    'category': candidate.get('category', 'N/A'),
                    'place_id': row.get('place_id', str(row.get('id', '')))
                }
                flat_final_candidates.append(flat_candidate)
            
            # Store recommendations in session state
            st.session_state.current_recommendations = flat_final_candidates
            
    elif method == "SVD-Based":
        st.subheader("SVD-Based Recommendations")
        svd_rec = SVDPlaceRecommender()
        svd_rec.evaluate_model(places)
        svd_rec.fit(places)
        recommendations = svd_rec.get_recommendations(places, user_lat, user_lng, predicted_ratings_dict, top_n=10, max_distance=5)
        # Store recommendations in session state
        st.session_state.current_recommendations = recommendations
        
    elif method == "Transfer-Based":
        st.subheader("Transfer-Based Recommendations")
        recommender = TransferBasedRecommender()
        recommendations = recommender.get_recommendations(user_lat, user_lng, user_preferences_dict, num_recs)
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
