import streamlit as st
import numpy as np
import pandas as pd
import joblib
import math
import pydeck as pdk
import networkx as nx
import openrouteservice
from openrouteservice import convert
from tensorflow.keras.models import load_model
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
import os

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

def show_map(recs, ors_key):
    """Display an interactive PyDeck map with an optimized route and text labels."""
    if not recs:
        return

    # Determine the data structure: nested ("row") or flat.
    if 'row' in recs[0]:
        names = [cand['row']['name'] for cand in recs]
        lats  = [cand['row']['lat'] for cand in recs]
        lons  = [cand['row']['lng'] for cand in recs]
    else:
        names = [cand['name'] for cand in recs]
        lats  = [cand['lat'] for cand in recs]
        lons  = [cand['lng'] for cand in recs]
    
    map_df = pd.DataFrame({
        'name': names,
        'lat': lats,
        'lon': lons
    })
    
    route_coords = None
    if not map_df.empty:
        # Try OpenRouteService routing if an API key is provided.
        if ors_key and ors_key != "":
            try:
                client = openrouteservice.Client(key=ors_key)
                coords = map_df[['lon', 'lat']].values.tolist()
                route_geojson = client.directions(coords, profile='driving-car', format='geojson')
                route_coords = route_geojson['features'][0]['geometry']['coordinates']
            except Exception as e:
                st.error(f"Routing API error: {e}")
                route_coords = None
        # Fallback: use TSP optimization with Euclidean distances.
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



# ---------------------------
# Load Resources
# ---------------------------
# Update load_resources to include type processing
@st.cache(allow_output_mutation=True)
def load_resources():
    base_path = "models"
    auto_model = load_model(os.path.join(base_path, "autoencoder.h5"))
    scaler = joblib.load(os.path.join(base_path, "scaler.save"))
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

# ---------------------------
# Base Recommendation Class
# ---------------------------
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
# SVD-Based Recommendation
# ---------------------------
class SVDRecommender(BaseRecommender):
    def __init__(self):
        self.model = SVD(n_factors=150, n_epochs=10, lr_all=0.002, reg_all=0.02)
        self.reader = Reader(rating_scale=(1, 5))
        
    def prepare_data(self):
        ratings = []
        for _, row in places.iterrows():
            if pd.notna(row['rating']):
                ratings.append({
                    'user': 'default_user',
                    'item': row['name'],
                    'rating': row['rating']
                })
        return Dataset.load_from_df(pd.DataFrame(ratings), self.reader)
    
    def get_recommendations(self, user_lat, user_lon, user_prefs, num_recs):
        data = self.prepare_data()
        trainset = data.build_full_trainset()
        self.model.fit(trainset)
        
        recommendations = []
        for _, row in places.iterrows():
            dist = haversine(user_lat, user_lon, row['lat'], row['lng'])
            if dist > 5000:
                continue
            
            try:
                pred = self.model.predict('default_user', row['name']).est
            except:
                pred = 3.0
                
            type_score = 0
            for i, cat in enumerate(categories):
                if any(pt in row['types_processed'] for pt in category_to_place_types.get(cat, [])):
                    type_score += user_prefs[i]
            type_score /= 5.0  # Normalize
            
            score = (0.4 * pred/5.0 +
                    0.4 * type_score +
                    0.2 * (1 - dist/5000))
            
            recommendations.append({
                'name': row['name'],
                'lat': row['lat'],
                'lng': row['lng'],
                'score': score,
                'pred_rating': pred,
                'type_score': type_score,
                'distance': dist
            })
        
        return sorted(recommendations, key=lambda x: x['score'], reverse=True)[:num_recs]

# ---------------------------
# UI Setup
# ---------------------------
st.sidebar.header("User Settings")
user_lat = st.sidebar.number_input("Latitude", value=40.4168)
user_lng = st.sidebar.number_input("Longitude", value=-3.7038)
num_recs = st.sidebar.slider("Number of Recommendations", 1, 20, 5)
method = st.sidebar.selectbox("Method", ["Autoencoder-Based", "SVD-Based"])

st.title("Place Recommendations")
st.write("Rate your preferences. Check a category to enable its rating.")

input_ratings = []
provided_mask = []
for cat in categories:
    provide = st.checkbox(f"Rate **{cat}**", key=f"chk_{cat}")
    if provide:
        rating = st.slider(f"Rating for {cat}:", min_value=0.0, max_value=5.0, step=0.5, key=f"slider_{cat}")
        input_ratings.append(rating)
        provided_mask.append(True)
    else:
        input_ratings.append(0.0)
        provided_mask.append(False)
input_ratings = np.array(input_ratings, dtype=np.float32).reshape(1, -1)
provided_mask = np.array(provided_mask, dtype=bool).reshape(1, -1)

if st.button("Generate Recommendations"):
    # Predict missing ratings using the autoencoder
    input_scaled = scaler.transform(input_ratings)
    predicted_scaled = auto_model.predict(input_scaled)
    predicted_scaled = np.clip(predicted_scaled, 0, 1)
    predicted = scaler.inverse_transform(predicted_scaled)
    final_predictions = predicted[0]
    # Overwrite predicted values with provided ratings
    final_predictions[provided_mask[0]] = input_ratings[0][provided_mask[0]]
    # Build predicted ratings dict for later use (if needed)
    predicted_ratings_dict = {cat: final_predictions[i] for i, cat in enumerate(categories)}
    
    # Now, call the autoencoder recommender with the provided_mask
    recommender = AutoencoderRecommender()
    recommendations = recommender.get_recommendations(user_lat, user_lng, final_predictions, provided_mask[0], num_recs)
    
    # Display recommendations and map (ensure recommendations have a 'row' key if your show_map expects it)
    for rec in recommendations:
        st.write(f"**{rec['name']}** (Score: {rec['score']:.2f})")
    show_map(recommendations, ors_key="")  # or pass your ORS API key if available