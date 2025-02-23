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
    map_df = pd.DataFrame({
        'name': [cand['row']['name'] for cand in recs],
        'lat': [cand['row']['lat'] for cand in recs],
        'lon': [cand['row']['lng'] for cand in recs]
    })
    route_coords = None
    if not map_df.empty:
        # If ORS API key is provided, try street-based routing.
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
        
        # Create a TextLayer for place names.
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
        # Create a PathLayer for the route.
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
    auto_model = load_model("autoencoder.h5")
    scaler = joblib.load("scaler.save")
    places = pd.read_csv("combined_places.csv")
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
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
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
        st.write("SVD Model Evaluation:")
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
# Sidebar: User Location, Routing Options, and Number of Recommendations
# ---------------------------
st.sidebar.header("Your Location")
user_lat = st.sidebar.number_input("Latitude", value=40.4168, format="%.4f")
user_lng = st.sidebar.number_input("Longitude", value=-3.7038, format="%.4f")

st.sidebar.header("Routing Options")
ors_key = st.sidebar.text_input("OpenRouteService API Key (optional)", value="", type="password")

st.sidebar.header("Recommendation Settings")
method = st.sidebar.selectbox("Method", ["Autoencoder-Based", "SVD-Based"])
num_recs = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=20, value=5, step=1)

# ---------------------------
# Main UI: User Preferences
# ---------------------------
st.title("Personalized Place Recommendations in Madrid")
st.write("Rate your preferences. Check a category and use the slider (0–5) for those you care about.")

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

# ---------------------------
# Generate Recommendations Button
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
    
    # Hideable expander for Predicted Category Ratings
    with st.expander("Show Predicted Category Ratings"):
        for cat, rating in predicted_ratings_dict.items():
            st.write(f"**{cat}:** {rating:.2f}")
    
    if method == "Autoencoder-Based":
        # --- Autoencoder-Based Recommendations ---
        candidates = []
        for idx, row in places.iterrows():
            dist = haversine(user_lat, user_lng, row['lat'], row['lng'])
            if dist > 2000:
                continue
            best_cat = None
            best_factor = 0
            for cat, pred_val in predicted_ratings_dict.items():
                mapped_types = category_to_place_types.get(cat, [])
                if any(pt in row['types'].lower() for pt in mapped_types):
                    normalized = pred_val / 5.0
                    if normalized > best_factor:
                        best_factor = normalized
                        best_cat = cat
            if best_cat is None:
                continue
            candidates.append({
                'row': row,
                'distance': dist,
                'pred_factor': best_factor,
                'category': best_cat
            })
        
        if not candidates:
            st.error("No candidate places found within 2 km.")
        else:
            max_reviews = max([row['user_ratings_total'] for candidate in candidates 
                               for row in [candidate['row']] if pd.notna(row['user_ratings_total'])] or [1])
            w_distance = 0.1
            w_rating = 0.2
            w_reviews = 0.2
            w_pred = 0.5
            for candidate in candidates:
                row = candidate['row']
                dist = candidate['distance']
                distance_score = 1 - (dist / 2000)
                norm_rating = (row['rating'] / 5.0) if pd.notna(row['rating']) else 0
                norm_reviews = (np.log(row['user_ratings_total'] + 1) / np.log(max_reviews + 1)) if pd.notna(row['user_ratings_total']) else 0
                candidate['score'] = (w_distance * distance_score +
                                      w_rating * norm_rating +
                                      w_reviews * norm_reviews +
                                      w_pred * candidate['pred_factor'])
            groups = {}
            for candidate in candidates:
                cat = candidate['category']
                groups.setdefault(cat, []).append(candidate)
            for cat in groups:
                groups[cat] = sorted(groups[cat], key=lambda x: x['score'], reverse=True)
            final_candidates = []
            round_idx = 0
            while len(final_candidates) < num_recs:
                added_this_round = False
                for cat, cand_list in groups.items():
                    if len(cand_list) > round_idx:
                        final_candidates.append(cand_list[round_idx])
                        added_this_round = True
                        if len(final_candidates) == num_recs:
                            break
                if not added_this_round:
                    break
                round_idx += 1
            final_candidates = sorted(final_candidates, key=lambda x: x['score'], reverse=True)
            
            st.subheader("Top Place Recommendations (Autoencoder-Based)")
            if final_candidates:
                best = final_candidates[0]
                best_row = best['row']
                # Use description if available; otherwise, fallback to types.
                best_desc = best_row.get('description', '')
                if not best_desc or pd.isna(best_desc) or best_desc.strip() == "":
                    best_desc = best_row['types']
                st.markdown("### Best Recommendation")
                st.image(best_row['icon'], width=80)
                st.write(f"**Name:** {best_row['name']}")
                st.write(f"**Category:** {best['category']}")
                st.write(f"**Distance:** {best['distance']:.0f} m")
                st.write(f"**Place Rating:** {best_row['rating']}")
                st.write(f"**Reviews:** {best_row['user_ratings_total']}")
                st.write(f"**Description:** {best_desc}")
                st.write(f"**Coordinates:** (Lat: {best_row['lat']}, Lon: {best_row['lng']})")
                st.write(f"**Score:** {best['score']:.2f}")
                if len(final_candidates) > 1:
                    st.markdown("### Other Recommendations")
                    for candidate in final_candidates[1:]:
                        row = candidate['row']
                        desc = row.get('description', '')
                        if not desc or pd.isna(desc) or desc.strip() == "":
                            desc = row['types']
                        with st.container():
                            cols = st.columns([1, 3])
                            with cols[0]:
                                st.image(row['icon'], width=60)
                            with cols[1]:
                                st.write(f"**Name:** {row['name']}")
                                st.write(f"Category: {candidate['category']}, Distance: {candidate['distance']:.0f} m")
                                st.write(f"Rating: {row['rating']}, Reviews: {row['user_ratings_total']}")
                                st.write(f"Description: {desc}")
                                st.write(f"Coordinates: (Lat: {row['lat']}, Lon: {row['lng']})")
                                st.write(f"Score: {candidate['score']:.2f}")
            else:
                st.error("No balanced recommendations found.")
            
            st.subheader("Map View: Recommended Places & Optimized Route")
            show_map(final_candidates, ors_key)
    
    else:
        # --- SVD-Based Recommendations ---
        st.subheader("SVD-Based Recommendations")
        svd_rec = SVDPlaceRecommender()
        svd_rec.evaluate_model(places)
        svd_rec.fit(places)
        recommendations = svd_rec.get_recommendations(places, user_lat, user_lng, predicted_ratings_dict, top_n=num_recs, max_distance=5)
        
        st.subheader("Top Place Recommendations (SVD-Based)")
        if recommendations:
            best = recommendations[0]
            best_desc = best.get('description', '')
            if not best_desc or pd.isna(best_desc) or best_desc.strip() == "":
                best_desc = best['types']
            st.markdown("### Best Recommendation")
            st.image(best['icon'], width=80)
            st.write(f"**Name:** {best['name']}")
            st.write(f"**Distance:** {best['distance']:.2f} km")
            st.write(f"**Place Rating:** {best['actual_rating']}")
            st.write(f"**Reviews:** {best['user_ratings_total']}")
            st.write(f"**Description:** {best_desc}")
            st.write(f"**Coordinates:** (Lat: {best['lat']}, Lon: {best['lng']})")
            st.write(f"**Score:** {best['score']:.2f}")
            st.markdown("### Other Recommendations")
            for rec in recommendations[1:]:
                desc = rec.get('description', '')
                if not desc or pd.isna(desc) or desc.strip() == "":
                    desc = rec['types']
                with st.container():
                    cols = st.columns([1, 3])
                    with cols[0]:
                        st.image(rec['icon'], width=60)
                    with cols[1]:
                        st.write(f"**Name:** {rec['name']}")
                        st.write(f"Distance: {rec['distance']:.2f} km, Type Score: {rec['type_score']:.2f}")
                        st.write(f"Rating: {rec['actual_rating']} ({int(rec['user_ratings_total'])} reviews)")
                        st.write(f"Description: {desc}")
                        st.write(f"Coordinates: (Lat: {rec['lat']}, Lon: {rec['lng']})")
                        st.write(f"Score: {rec['score']:.2f}")
        else:
            st.error("No recommendations found with SVD-based method.")
        
        st.subheader("Map View: Recommended Places & Optimized Route")
        map_df = pd.DataFrame({
            'name': [rec['name'] for rec in recommendations],
            'lat': [rec['lat'] for rec in recommendations],
            'lon': [rec['lng'] for rec in recommendations]
        })
        if map_df.empty:
            st.error("No map data available for SVD recommendations.")
        else:
            recs_for_map = [{'row': rec} for rec in recommendations]
            show_map(recs_for_map, ors_key)