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
# Load Resources (Model, Scaler, Places Data)
# ---------------------------
@st.cache(allow_output_mutation=True)
def load_resources():
    model = load_model("autoencoder.h5")
    scaler = joblib.load("scaler.save")
    places = pd.read_csv("combined_places.csv")
    places['types_processed'] = places['types'].apply(process_types)
    return model, scaler, places

model, scaler, places = load_resources()

# ---------------------------
# Sidebar: User Location & Routing Options
# ---------------------------
st.sidebar.header("Your Location")
user_lat = st.sidebar.number_input("Latitude", value=40.4168, format="%.4f")
user_lng = st.sidebar.number_input("Longitude", value=-3.7038, format="%.4f")

st.sidebar.header("Routing Options")
ors_key = st.sidebar.text_input("OpenRouteService API Key (optional)", value="", type="password")

# ---------------------------
# Main UI: Rate Your Preferences
# ---------------------------
st.title("Personalized Place Recommendations in Madrid")
st.write("Rate your preferences. Check a category and use the slider (0–5) for the ones you care about.")

input_ratings = []
provided_mask = []
for cat in categories:
    rating = st.slider(f"Rating for {cat} (0=dislike, 5=like)", 0.0, 5.0, 0.0)
    input_ratings.append(rating)
    provided_mask.append(True)

input_ratings = np.array(input_ratings, dtype=np.float32).reshape(1, -1)
provided_mask = np.array(provided_mask, dtype=bool).reshape(1, -1)

# ---------------------------
# Generate Recommendations Button
# ---------------------------
if st.button("Generate Recommendations"):
    # 1. Predict Full Ratings using the Autoencoder
    input_scaled = scaler.transform(input_ratings)
    predicted_scaled = model.predict(input_scaled)
    predicted_scaled = np.clip(predicted_scaled, 0, 1)
    predicted = scaler.inverse_transform(predicted_scaled)
    final_predictions = predicted[0]
    final_predictions[provided_mask[0]] = input_ratings[0][provided_mask[0]]
    predicted_ratings_dict = {cat: final_predictions[i] for i, cat in enumerate(categories)}

    st.subheader("Predicted Category Ratings")
    for cat, rating in predicted_ratings_dict.items():
        st.write(f"**{cat}:** {rating:.2f}")

    # 2. Candidate Selection from Places Data (within 2 km)
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
            'row': row,
            'distance': dist,
            'pred_factor': best_factor,
            'category': best_cat
        })

    if not candidates:
        st.error("No candidate places found within 2 km.")
    else:
        # 3. Compute Composite Scores
        max_reviews = max([row['user_ratings_total'] for candidate in candidates
                           for row in [candidate['row']] if pd.notna(row['user_ratings_total'])] or [1])
        w_distance = 0.1
        w_rating = 0.25
        w_reviews = 0.25
        w_pred = 0.4
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

        # 4. Balanced Candidate Selection (Round-Robin)
        groups = {}
        for candidate in candidates:
            cat = candidate['category']
            groups.setdefault(cat, []).append(candidate)
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

        # 5. Display Recommendations with Photos
        st.subheader("Top Place Recommendations")
        if final_candidates:
            best = final_candidates[0]
            best_row = best['row']
            st.markdown("### Best Recommendation")
            st.image(best_row['icon'], width=80)
            st.write(f"**Name:** {best_row['name']}")
            st.write(f"**Category:** {best['category']}")
            st.write(f"**Distance:** {best['distance']:.0f} m")
            st.write(f"**Place Rating:** {best_row['rating']}")
            st.write(f"**Reviews:** {best_row['user_ratings_total']}")
            st.write(f"**Coordinates:** (Lat: {best_row['lat']}, Lon: {best_row['lng']})")
            st.write(f"**Score:** {best['score']:.2f}")

            if len(final_candidates) > 1:
                st.markdown("### Other Recommendations")
                for candidate in final_candidates[1:]:
                    row = candidate['row']
                    with st.container():
                        cols = st.columns([1, 3])
                        with cols[0]:
                            st.image(row['icon'], width=60)
                        with cols[1]:
                            st.write(f"**Name:** {row['name']}")
                            st.write(f"Category: {candidate['category']}, Distance: {candidate['distance']:.0f} m")
                            st.write(f"Rating: {row['rating']}, Reviews: {row['user_ratings_total']}")
                            st.write(f"Coordinates: (Lat: {row['lat']}, Lon: {row['lng']})")
                            st.write(f"Score: {candidate['score']:.2f}")
        else:
            st.error("No balanced recommendations found.")

        # 6. Map Visualization with Optimized Route and Text Labels
        st.subheader("Map View: Recommended Places & Optimized Route")
        map_df = pd.DataFrame({
            'name': [cand['row']['name'] for cand in final_candidates],
            'lat': [cand['row']['lat'] for cand in final_candidates],
            'lon': [cand['row']['lng'] for cand in final_candidates]
        })

        if not map_df.empty:
            # Use ORS routing if API key is provided.
            route_coords = None
            if ors_key and ors_key != "":
                try:
                    client = openrouteservice.Client(key=ors_key)
                    coords = map_df[['lon', 'lat']].values.tolist()
                    route_geojson = client.directions(coords, profile='driving-car', format='geojson')
                    route_coords = route_geojson['features'][0]['geometry']['coordinates']
                except Exception as e:
                    st.error(f"Routing API error: {e}")
                    route_coords = None
            # Fallback: use TSP optimization based on Euclidean distances.
            if route_coords is None:
                G = nx.complete_graph(len(map_df))
                coords = map_df[['lat', 'lon']].values
                for i, j in G.edges():
                    G[i][j]['weight'] = euclidean_distance((coords[i][0], coords[i][1]), (coords[j][0], coords[j][1]))
                tsp_route = nx.approximation.traveling_salesman_problem(G, weight='weight')
                optimized_data = map_df.iloc[tsp_route].reset_index(drop=True)
                route_coords = optimized_data[['lon', 'lat']].values.tolist()
            else:
                optimized_data = map_df.copy()

            # Create a PathLayer for the route.
            path_layer = pdk.Layer(
                "PathLayer",
                data=[{"path": route_coords}],
                get_path="path",
                get_color="[0, 0, 255]",
                width_scale=20,
                width_min_pixels=3,
            )
            # Create a TextLayer to display place names as "signs" on the map.
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
