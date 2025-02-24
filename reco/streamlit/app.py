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

# For SVD-based recommendations:
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

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
    # Determine if the recs are nested (with key 'row') or flat.
    if not recs:
        return
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
        if ors_key and ors_key != "":
            try:
                client = openrouteservice.Client(key=ors_key)
                coords = map_df[['lon', 'lat']].values.tolist()
                route_geojson = client.directions(coords, profile='driving-car', format='geojson')
                route_coords = route_geojson['features'][0]['geometry']['coordinates']
            except Exception as e:
                st.error(f"Routing API error: {e}")
                route_coords = None
        # Fallback: TSP optimization using Euclidean distances.
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
    
    #st.write("DEBUG: Recommendation Data:", rec)
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
# ---------------------------
# Constants and Mappings
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
# Load Resources (Autoencoder Model, Scaler, Places Data)
# ---------------------------
@st.cache(allow_output_mutation=True)
def load_resources():
    auto_model = load_model("models/autoencoder.h5")
    scaler = joblib.load("models/scaler.save")
    places = pd.read_csv("resources/combined_places.csv")
    places['types_processed'] = places['types'].apply(process_types)
    return auto_model, scaler, places

auto_model, scaler, places = load_resources()

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
# (Assumes TransferRecommender is imported from your src folder)
from src.transfer_recommender import TransferRecommender
class TransferBasedRecommender:
    def __init__(self):
        self.tr = TransferRecommender()
        self.tr.train_base_model(save_path="models")
        self.places = pd.read_csv("resources/combined_places.csv")
        self.places['types_processed'] = self.places['types'].apply(process_types)
        self.tr.transfer_to_places(self.places, save_path="models")
    
    def get_recommendations(self, user_lat, user_lon, user_prefs, num_recs):
        return self.tr.get_recommendations(
            user_preferences=user_prefs,
            user_lat=user_lat,
            user_lon=user_lon,
            places_df=self.places,
            top_n=num_recs
            
        )

# ---------------------------
# UI Setup: Sidebar and Main Input
# ---------------------------
st.sidebar.header("User Settings")
user_lat = st.sidebar.number_input("Latitude", value=40.4168, format="%.4f")
user_lng = st.sidebar.number_input("Longitude", value=-3.7038, format="%.4f")
#num_recs = st.sidebar.slider("Number of Recommendations", 1, 20, 5)
ors_key = st.sidebar.text_input("OpenRouteService API Key (optional)", value="", type="password")
method = st.sidebar.selectbox("Method", ["Autoencoder-Based", "SVD-Based", "Transfer-Based"])
num_recs = 5
st.title("Personalized Place Recommendations in Madrid")
st.write("Rate your preferences. Check a category and use the slider (0–5) for those you care about.")

# Capture user preferences with checkboxes and sliders
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

# Also prepare a dictionary for Transfer-Based preferences if needed
user_preferences_dict = {cat: float(st.session_state.get(f"slider_{cat}", 0.0)) for cat in categories}

# ---------------------------
# Generate Recommendations
# ---------------------------
if st.button("Generate Recommendations"):
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
        # --- Autoencoder-Based Recommendations ---
        candidates = []
        for idx, row in places.iterrows():
            dist = haversine(user_lat, user_lng, row['lat'], row['lng'])
            if dist > 2000:
                continue
            best_cat = None
            best_factor = 0
            # Compare using processed types for matching
            for cat, pred_val in predicted_ratings_dict.items():
                mapped_types = category_to_place_types.get(cat, [])
                if any(pt in row['types_processed'] for pt in mapped_types):
                    normalized = pred_val / 5.0
                    if normalized > best_factor:
                        best_factor = normalized
                        best_cat = cat
            if best_cat is None:
                continue
            # Convert row to dictionary so keys are accessible via .get()
            candidates.append({
                'row': row.to_dict(),
                'distance': dist,
                'pred_factor': best_factor,
                'category': best_cat
            })
        if not candidates:
            st.error("No candidate places found within 2 km.")
        else:
            # Compute normalization values
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
            # Group candidates by category for balanced selection
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
            
            # Flatten candidate dictionaries so they match the SVD logic:
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
                    'category': candidate.get('category', 'N/A')
                }
                flat_final_candidates.append(flat_candidate)
            
            st.subheader("Top Place Recommendations (Autoencoder-Based)")
            if flat_final_candidates:
                st.markdown("### Best Recommendation")
                display_recommendation(flat_final_candidates[0])
                st.markdown("### Other Recommendations")
                for rec in flat_final_candidates[1:]:
                    display_recommendation(rec)
            else:
                st.error("No balanced recommendations found.")
            
            st.subheader("Map View: Recommended Places & Optimized Route")
            # Wrap flat dictionaries in a 'row' key for mapping (as expected by show_map)
            recs_for_map = [{'row': rec} for rec in flat_final_candidates]
            show_map(recs_for_map, ors_key)
    
    elif method == "SVD-Based":
        st.subheader("SVD-Based Recommendations")
        svd_rec = SVDPlaceRecommender()
        svd_rec.evaluate_model(places)
        svd_rec.fit(places)
        recommendations = svd_rec.get_recommendations(places, user_lat, user_lng, predicted_ratings_dict, top_n=10, max_distance=5)
        
        st.subheader("Top Place Recommendations (SVD-Based)")
        if recommendations:
            # Display the best recommendation prominently
            best = recommendations[0]
            st.markdown("### Best Recommendation")
            display_recommendation(best)
            st.markdown("### Other Recommendations")
            for rec in recommendations[1:]:
                display_recommendation(rec)
        else:
            st.error("No recommendations found with SVD-based method.")
        
        st.subheader("Map View: Recommended Places & Optimized Route")
        # For SVD recommendations, adjust the data structure for mapping if needed
        recs_for_map = [{'row': rec} for rec in recommendations]
        show_map(recs_for_map, ors_key)
    
    elif method == "Transfer-Based":
        st.subheader("Transfer-Based Recommendations")
        recommender = TransferBasedRecommender()
        recommendations = recommender.get_recommendations(user_lat, user_lng, user_preferences_dict, num_recs)
        if recommendations:
            for rec in recommendations:
                display_recommendation(rec)
        else:
            st.error("No recommendations found with Transfer-Based method.")
        
        st.subheader("Map View: Recommended Places & Optimized Route")
        show_map(recommendations, ors_key)