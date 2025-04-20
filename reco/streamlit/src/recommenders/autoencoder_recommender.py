"""
Autoencoder-based recommendation model.
"""

import numpy as np
import pandas as pd
import math
from .base_recommender import BaseRecommender
import joblib

# Assuming these are defined elsewhere and will be passed to the recommender
# from the app.py initialization
auto_model = None
scaler = None
places = None
categories = None
category_to_place_types = None

class AutoencoderRecommender(BaseRecommender):
    """
    A recommender that uses an autoencoder neural network to learn
    user preferences and recommend places.
    """
    
    def __init__(self, auto_model=None, scaler=None, places_df=None, 
                 categories_list=None, category_mappings=None):
        """
        Initialize the autoencoder recommender.
        
        Args:
            auto_model: The trained autoencoder model
            scaler: The scaler used for normalizing the inputs
            places_df: DataFrame containing place data
            categories_list: List of categories
            category_mappings: Mapping from categories to place types
        """
        self.auto_model = auto_model
        self.scaler = scaler
        self.places_df = places_df
        self.categories = categories_list
        self.category_to_place_types = category_mappings
        
    def haversine(self, lat1, lon1, lat2, lon2):
        """Compute distance (in meters) between two lat/lon points."""
        R = 6371000  # Earth's radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c

    def get_recommendations(self, user_lat, user_lon, user_prefs, provided_mask, num_recs=5):
        """
        Get place recommendations based on user preferences and location.
        
        Args:
            user_lat: User's latitude
            user_lon: User's longitude
            user_prefs: Array of user preference ratings
            provided_mask: Mask indicating which preferences were explicitly provided
            num_recs: Number of recommendations to return
            
        Returns:
            list: Recommended places
        """
        # Use global model and data if not provided during initialization
        _auto_model = self.auto_model or auto_model
        _scaler = self.scaler or scaler
        _places_df = self.places_df or places
        _categories = self.categories or categories
        _category_to_place_types = self.category_to_place_types or category_to_place_types
        
        if _auto_model is None or _scaler is None or _places_df is None or _categories is None:
            raise ValueError("Model, scaler, places data, and categories must be provided")
            
        # Reconstruct predicted ratings
        input_scaled = _scaler.transform([user_prefs])
        predicted_scaled = _auto_model.predict(input_scaled)
        predicted = _scaler.inverse_transform(np.clip(predicted_scaled, 0, 1))[0]
        final_predictions = np.where(provided_mask, user_prefs, predicted)

        # Candidate scoring logic
        candidates = []
        for _, row in _places_df.iterrows():
            dist = self.haversine(user_lat, user_lon, row['lat'], row['lng'])
            if dist > 2000:  # Max distance 2km
                continue

            best_cat = None
            best_score = 0
            for i, cat in enumerate(_categories):
                mapped_types = _category_to_place_types.get(cat, [])
                if any(pt in row['types_processed'] for pt in mapped_types):
                    cat_score = final_predictions[i] / 5.0
                    if cat_score > best_score:
                        best_score = cat_score
                        best_cat = cat

            if best_cat:
                norm_rating = (row['rating'] / 5.0) if pd.notna(row['rating']) else 0
                norm_reviews = (np.log(row['user_ratings_total'] + 1)) / np.log(_places_df['user_ratings_total'].max() + 1) if pd.notna(row['user_ratings_total']) else 0
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
            'reviews': r['row']['user_ratings_total'],
            'place_id': r['row'].get('place_id', ''),
            'types': r['row'].get('types', ''),
            'vicinity': r['row'].get('vicinity', ''),
            'icon': r['row'].get('icon', ''),
            'user_ratings_total': r['row']['user_ratings_total'],
            'distance': r['distance']
        } for r in final_recs[:num_recs]] 