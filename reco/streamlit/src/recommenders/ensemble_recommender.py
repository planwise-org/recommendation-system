"""
Ensemble recommender that combines multiple recommendation models.
"""

import numpy as np
import math
from .base_recommender import BaseRecommender
from .autoencoder_recommender import AutoencoderRecommender
from .svd_recommender import SVDPlaceRecommender
from .transfer_recommender import TransferRecommender

class EnsembleRecommender(BaseRecommender):
    """
    A recommender that combines results from multiple recommendation models to provide
    better and more diverse place recommendations.

    This ensemble approach uses three different recommendation algorithms:
    - Autoencoder-Based: Uses neural network to learn user preferences and find similar places
    - SVD-Based: Uses matrix factorization for collaborative filtering
    - Transfer Learning-Based: Leverages knowledge from movie domain to recommend places

    The ensemble works by:
    1. Getting recommendations from each individual model
    2. Normalizing scores across models to make them comparable
    3. Combining recommendations with weighted scores based on model weights
    4. De-duplicating places that appear in multiple recommendation sets
    5. Ranking based on the combined ensemble score
    """

    def __init__(self, weights=None):
        """
        Initialize the ensemble recommender with optional weights for each model.

        Args:
            weights (dict, optional): Dictionary with weights for each model.
                Example: {'autoencoder': 0.4, 'svd': 0.3, 'transfer': 0.3}
                If None, equal weights will be used.
        """
        self.weights = weights or {'autoencoder': 0.33, 'svd': 0.33, 'transfer': 0.34}
        self.autoencoder_recommender = None
        self.svd_recommender = None
        self.transfer_recommender = None
        self.places_df = None
        self.max_distance = 2000  # meters
        self.category_to_place_types = None

    def initialize_models(self, auto_model, scaler, places_df, category_to_place_types):
        """
        Initialize all the recommender models.

        Args:
            auto_model: The autoencoder model
            scaler: The scaler used for autoencoder input/output
            places_df: DataFrame containing places data
            category_to_place_types: Mapping from categories to place types
        """
        self.places_df = places_df
        self.category_to_place_types = category_to_place_types

        # Initialize component recommenders
        self.autoencoder_recommender = AutoencoderRecommender(
            auto_model=auto_model,
            scaler=scaler,
            places_df=places_df,
            category_mappings=category_to_place_types
        )

        self.svd_recommender = SVDPlaceRecommender(
            category_to_place_types=category_to_place_types
        )
        self.svd_recommender.fit(places_df)

        self.transfer_recommender = TransferRecommender()
        self.transfer_recommender.train_base_model()
        self.transfer_recommender.transfer_to_places(places_df)

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

    def _normalize_scores(self, recommendations):
        """Normalize scores to be between 0 and 1."""
        if not recommendations:
            return []

        # Get the min and max scores
        scores = [rec.get('score', 0) for rec in recommendations]
        min_score = min(scores)
        max_score = max(scores)

        # If all scores are the same, set normalized_score to 1.0 for all items
        if max_score == min_score:
            for rec in recommendations:
                rec['normalized_score'] = 1.0
            return recommendations

        # Normalize each score
        for rec in recommendations:
            if 'score' in rec:
                rec['normalized_score'] = (rec['score'] - min_score) / (max_score - min_score)
            else:
                rec['normalized_score'] = 0

        return recommendations

    def get_recommendations(self, user_lat, user_lon, user_prefs, predicted_ratings_dict, num_recs=5):
        """
        Get recommendations by combining results from all three models.

        Args:
            user_lat: User's latitude
            user_lon: User's longitude
            user_prefs: User preferences array or dictionary
            predicted_ratings_dict: Dictionary of predicted category ratings
            num_recs: Number of recommendations to return

        Returns:
            list: Combined and ranked recommendations
        """
        # Get recommendations from each model
        auto_recs = self._get_autoencoder_recommendations(user_lat, user_lon, user_prefs, num_recs*2)
        svd_recs = self._get_svd_recommendations(user_lat, user_lon, predicted_ratings_dict, num_recs*2)
        transfer_recs = self._get_transfer_recommendations(user_lat, user_lon, predicted_ratings_dict, num_recs*2)

        # Convert all recommendations to a consistent format and normalize scores
        auto_recs = self._normalize_scores(auto_recs)
        svd_recs = self._normalize_scores(svd_recs)
        transfer_recs = self._normalize_scores(transfer_recs)

        # Assign source information
        for rec in auto_recs:
            rec['source'] = 'autoencoder'
        for rec in svd_recs:
            rec['source'] = 'svd'
        for rec in transfer_recs:
            rec['source'] = 'transfer'

        # Combine all recommendations
        all_recs = auto_recs + svd_recs + transfer_recs

        # Create a dictionary to deduplicate by place_id or name
        unique_recs = {}
        for rec in all_recs:
            place_id = rec.get('place_id', rec.get('name', ''))

            if place_id in unique_recs:
                # Update with weighted score if this place is already in our results
                existing_rec = unique_recs[place_id]
                existing_weight = self.weights.get(existing_rec['source'], 0.33)
                current_weight = self.weights.get(rec['source'], 0.33)

                # Combine scores based on weights
                total_weight = existing_weight + current_weight
                existing_rec['ensemble_score'] = (
                    (existing_rec.get('ensemble_score', existing_rec.get('normalized_score', 0)) * existing_weight) +
                    (rec.get('normalized_score', 0) * current_weight)
                ) / total_weight

                # Update sources to reflect that multiple models recommended this place
                existing_rec['sources'] = existing_rec.get('sources', [existing_rec['source']]) + [rec['source']]

            else:
                # Add new recommendation
                rec['ensemble_score'] = rec.get('normalized_score', 0) * self.weights.get(rec['source'], 0.33)
                rec['sources'] = [rec['source']]
                rec['explanation'] = f"Recommended primarily by {rec['source'].capitalize()} model with a weight of {self.weights.get(rec['source'], 0.33):.2f}"
                unique_recs[place_id] = rec

        # Convert back to list and sort by ensemble score
        final_recs = list(unique_recs.values())

        # Update explanations for places with multiple sources
        for rec in final_recs:
            if len(rec['sources']) > 1:
                source_weights = [f"{s.capitalize()} ({self.weights.get(s, 0.33):.2f})" for s in rec['sources']]
                rec['explanation'] = f"Recommended by multiple models: {', '.join(source_weights)}"

        final_recs.sort(key=lambda x: x.get('ensemble_score', 0), reverse=True)

        # Return the top N recommendations
        return final_recs[:num_recs]

    def _get_autoencoder_recommendations(self, user_lat, user_lon, user_prefs, num_recs):
        """Get recommendations from the autoencoder model."""
        if self.autoencoder_recommender is None:
            return []

        try:
            # Convert user_prefs dict to array if needed
            if isinstance(user_prefs, dict):
                # Placeholder code - you would need to implement this conversion
                # based on your actual preference structure
                provided_mask = np.array([True if val > 0 else False for val in user_prefs.values()])
                user_prefs_array = np.array(list(user_prefs.values()))
            else:
                # Assuming user_prefs is already an array and provided_mask is included
                user_prefs_array = user_prefs
                provided_mask = np.ones_like(user_prefs_array, dtype=bool)

            return self.autoencoder_recommender.get_recommendations(
                user_lat, user_lon, user_prefs_array, provided_mask, num_recs
            )
        except Exception as e:
            print(f"Error in autoencoder recommendations: {e}")
            return []

    def _get_svd_recommendations(self, user_lat, user_lon, predicted_ratings_dict, num_recs):
        """Get recommendations from the SVD model."""
        if self.svd_recommender is None:
            return []

        try:
            return self.svd_recommender.get_recommendations(
                self.places_df, user_lat, user_lon,
                predicted_ratings_dict, top_n=num_recs
            )
        except Exception as e:
            print(f"Error in SVD recommendations: {e}")
            return []

    def _get_transfer_recommendations(self, user_lat, user_lon, predicted_ratings_dict, num_recs):
        """Get recommendations from the transfer learning model."""
        if self.transfer_recommender is None:
            return []

        try:
            return self.transfer_recommender.get_recommendations(
                user_preferences=predicted_ratings_dict,
                user_lat=user_lat,
                user_lon=user_lon,
                places_df=self.places_df,
                top_n=num_recs
            )
        except Exception as e:
            print(f"Error in transfer recommendations: {e}")
            return []
