import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk
import networkx as nx
import openrouteservice
from transfer_recommender import TransferRecommender
import os

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

# ---------------------------
# Load Data and Initialize Recommender
# ---------------------------
@st.cache_resource
def initialize_recommender():
    recommender = TransferRecommender()
    recommender.train_base_model()
    return recommender

@st.cache_data
def load_places_data():
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the data file
    file_path = os.path.join(current_dir, "combined_places.csv")
    places = pd.read_csv(file_path)
    return places

recommender = initialize_recommender()
places = load_places_data()
recommender.transfer_to_places(places)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Personalized Place Recommendations in Madrid")

# Sidebar: User Location & Routing Options
st.sidebar.header("Your Location")
user_lat = st.sidebar.number_input("Latitude", value=40.4168, format="%.4f")
user_lng = st.sidebar.number_input("Longitude", value=-3.7038, format="%.4f")
ors_key = st.sidebar.text_input("OpenRouteService API Key (optional)", value="", type="password")

# Main UI: Rate Your Preferences
st.write("Rate your preferences (0-5) for the categories you care about:")

# Create user preferences dictionary from sliders
user_preferences = {}
for category in categories:
    rating = st.slider(f"Rating for {category}", 0.0, 5.0, 0.0)
    user_preferences[category] = rating

# Generate Recommendations Button
if st.button("Generate Recommendations"):
    recommendations = recommender.get_recommendations(
        user_preferences=user_preferences,
        user_lat=user_lat,
        user_lon=user_lng,
        places_df=places,
        top_n=5
    )
    
    # Display Recommendations
    st.subheader("Top Place Recommendations")
    if recommendations:
        # Display best recommendation
        best = recommendations[0]
        st.markdown("### Best Recommendation")
        if 'icon' in best:
            st.image(best['icon'], width=80)
        st.write(f"**Name:** {best['name']}")
        st.write(f"**Distance:** {best['distance']:.2f} km")
        if 'rating' in best and best['rating']:
            st.write(f"**Place Rating:** {best['rating']}")
        st.write(f"**Similarity Score:** {best['similarity']:.2f}")
        st.write(f"**Final Score:** {best['score']:.2f}")

        # Display other recommendations
        if len(recommendations) > 1:
            st.markdown("### Other Recommendations")
            for rec in recommendations[1:]:
                with st.container():
                    cols = st.columns([1, 3])
                    with cols[0]:
                        if 'icon' in rec:
                            st.image(rec['icon'], width=60)
                    with cols[1]:
                        st.write(f"**Name:** {rec['name']}")
                        st.write(f"Distance: {rec['distance']:.2f} km")
                        if 'rating' in rec and rec['rating']:
                            st.write(f"Rating: {rec['rating']}")
                        st.write(f"Similarity Score: {rec['similarity']:.2f}")
                        st.write(f"Final Score: {rec['score']:.2f}")

        # Map Visualization
        st.subheader("Map View: Recommended Places & Optimized Route")
        map_df = pd.DataFrame({
            'name': [rec['name'] for rec in recommendations],
            'lat': [rec['lat'] for rec in recommendations],
            'lon': [rec['lng'] for rec in recommendations]
        })

        if not map_df.empty:
            # Route optimization code (unchanged)
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

            if route_coords is None:
                G = nx.complete_graph(len(map_df))
                coords = map_df[['lat', 'lon']].values
                for i, j in G.edges():
                    G[i][j]['weight'] = np.hypot(coords[i][0]-coords[j][0], coords[i][1]-coords[j][1])
                tsp_route = nx.approximation.traveling_salesman_problem(G, weight='weight')
                optimized_data = map_df.iloc[tsp_route].reset_index(drop=True)
                route_coords = optimized_data[['lon', 'lat']].values.tolist()
            else:
                optimized_data = map_df.copy()

            # Create map layers
            path_layer = pdk.Layer(
                "PathLayer",
                data=[{"path": route_coords}],
                get_path="path",
                get_color="[0, 0, 255]",
                width_scale=20,
                width_min_pixels=3,
            )

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
