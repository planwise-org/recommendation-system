"""
Ensemble recommender that combines multiple recommendation models.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from .base_recommender import BaseRecommender
from .autoencoder_recommender import AutoencoderRecommender
from .svd_recommender import SVDPlaceRecommender
from .transfer_recommender import TransferRecommender
from .madrid_transfer_recommender import MadridTransferRecommender

class EnsembleRecommender(BaseRecommender):
    """
    An ensemble recommender that combines recommendations from multiple models
    to provide a balanced set of recommendations.
    """
    
    def __init__(self, weights=None):
        """
        Initialize the ensemble recommender.
        
        Args:
            weights (dict): Model weights for the ensemble
        """
        # Default weights that balance all recommendation approaches
        self.weights = weights or {
            'autoencoder': 0.20,  # Good for general preferences
            'svd': 0.20,          # Good for popular, highly-rated places
            'transfer': 0.30,     # Good for preference transfer from other domains
            'madrid_transfer': 0.30  # Good for specialized Madrid recommendations
        }
        
        # Initialize recommender instances
        self.autoencoder_recommender = None
        self.svd_recommender = None
        self.transfer_recommender = None
        self.madrid_transfer_recommender = None
        
        # Track unique recommendations to ensure diversity
        self.recommended_categories = defaultdict(int)
        self.distance_penalty_factor = 0.1  # Lower penalty for distance to allow further places
        
    def initialize_models(self, auto_model, scaler, places_df, category_to_place_types):
        """
        Initialize all component recommenders.
        
        Args:
            auto_model: Trained autoencoder model
            scaler: Data scaler for autoencoder
            places_df: DataFrame of places data
            category_to_place_types: Mapping of categories to place types
        """
        # Initialize autoencoder-based recommender
        self.autoencoder_recommender = AutoencoderRecommender(
            auto_model=auto_model,
            scaler=scaler,
            places_df=places_df,
            categories_list=list(category_to_place_types.keys()),
            category_mappings=category_to_place_types
        )
        
        # Initialize SVD-based recommender
        self.svd_recommender = SVDPlaceRecommender(
            category_to_place_types=category_to_place_types
        )
        self.svd_recommender.fit(places_df)
        
        # Initialize transfer learning-based recommender
        self.transfer_recommender = TransferRecommender()
        self.transfer_recommender.train_base_model()
        self.transfer_recommender.transfer_to_places(places_df)
        
        # Initialize Madrid transfer recommender with correct parameters
        self.madrid_transfer_recommender = MadridTransferRecommender(
            embedding_model_name='all-MiniLM-L6-v2',
            embedding_path='models/madrid_place_embeddings.npz'
        )
        # No need to train this model as it uses pre-computed embeddings
    
    def _normalize_scores(self, recommendations, min_score=0.0, max_score=1.0):
        """
        Normalize scores across all recommendations to a common range.
        
        Args:
            recommendations: List of recommendation dictionaries
            min_score: Minimum score value
            max_score: Maximum score value
            
        Returns:
            List of recommendations with normalized scores
        """
        if not recommendations:
            return []
            
        # Check if recommendations have a 'score' key
        # Some recommenders might use 'similarity' or other keys instead
        if 'score' not in recommendations[0]:
            # Try common alternative keys
            score_key = None
            for key in ['similarity', 'predicted_rating', 'distance_score']:
                if key in recommendations[0]:
                    score_key = key
                    break
                
            # If no score-like key is found, add a default score
            if score_key is None:
                for rec in recommendations:
                    rec['score'] = 0.5  # Default middle score
                return recommendations
            
            # Use the alternative key as the score
            for rec in recommendations:
                rec['score'] = rec[score_key]
        
        # Now normalize the scores
        scores = [rec['score'] for rec in recommendations]
        score_min, score_max = min(scores), max(scores)
        
        # Avoid division by zero
        score_range = score_max - score_min
        if score_range == 0:
            normalized_scores = [0.5 for _ in scores]
        else:
            normalized_scores = [(s - score_min) / score_range * (max_score - min_score) + min_score for s in scores]
        
        # Update scores in recommendations
        for i, rec in enumerate(recommendations):
            rec['original_score'] = rec['score']
            rec['score'] = normalized_scores[i]
            
        return recommendations
    
    def _apply_diversity_boost(self, recommendations, max_per_category=2):
        """
        Apply a diversity boost to ensure variety in recommendations.
        
        Args:
            recommendations: List of recommendation dictionaries
            max_per_category: Maximum number of places per category
            
        Returns:
            List of recommendations with diversity boost applied
        """
        category_counts = defaultdict(int)
        
        for rec in recommendations:
            category = rec.get('category', 'unknown')
            category_counts[category] += 1
            
            # Apply a boost based on category rarity
            if category_counts[category] <= max_per_category:
                rec['score'] *= (1.1 - 0.1 * (category_counts[category] - 1))
            else:
                # Penalty for over-represented categories
                rec['score'] *= 0.7
                
        return recommendations
    
    def _apply_distance_penalty(self, recommendations, max_distance=5.0):
        """
        Apply a penalty based on distance, but with a gentle curve to allow 
        some farther places if they're truly exceptional.
        
        Args:
            recommendations: List of recommendation dictionaries
            max_distance: Maximum distance in km before heavier penalties
            
        Returns:
            List of recommendations with distance penalty applied
        """
        for rec in recommendations:
            # Convert distance to km if it's in meters
            distance_km = rec.get('distance', 0) / 1000 if rec.get('distance', 0) > 100 else rec.get('distance', 0)
            
            # Sigmoid-like penalty that increases more rapidly after max_distance
            if distance_km <= 2.0:
                # Very minor penalty for places within 2km
                distance_factor = 1.0 - (distance_km * 0.05)
            elif distance_km <= max_distance:
                # Moderate penalty between 2km and max_distance
                distance_factor = 0.9 - ((distance_km - 2.0) * 0.1)
            else:
                # Stronger penalty beyond max_distance
                distance_factor = 0.6 - ((distance_km - max_distance) * 0.2)
            
            # Ensure the factor doesn't go below 0.1
            distance_factor = max(0.1, distance_factor)
            
            # Apply the distance penalty to the score
            rec['distance_factor'] = distance_factor
            rec['score'] *= distance_factor
            
        return recommendations
    
    def _apply_novelty_boost(self, recommendations):
        """
        Apply a boost to places that are less common in recommendations.
        
        Args:
            recommendations: List of recommendation dictionaries
            
        Returns:
            List of recommendations with novelty boost applied
        """
        # Apply a small boost to places with fewer reviews (discovery factor)
        for rec in recommendations:
            reviews = rec.get('user_ratings_total', 0) or 0
            
            # Novelty boost decreases as review count increases
            if reviews < 10:
                novelty_boost = 1.25  # Big boost for very undiscovered places
            elif reviews < 50:
                novelty_boost = 1.15  # Moderate boost
            elif reviews < 100:
                novelty_boost = 1.05  # Small boost
            else:
                novelty_boost = 1.0   # No boost for very popular places
                
            rec['novelty_boost'] = novelty_boost
            rec['score'] *= novelty_boost
            
        return recommendations
    
    def _standardize_recommendation(self, rec, source):
        """
        Standardize recommendation format from different recommenders.
        
        Args:
            rec: Recommendation dictionary from a component recommender
            source: Source recommender name
            
        Returns:
            Standardized recommendation dictionary
        """
        # Extract place_id - some recommenders have it directly, others within 'row'
        place_id = rec.get('place_id', '')
        if not place_id and 'row' in rec:
            place_id = rec['row'].get('place_id', '')
            
        # Extract category - sometimes it's a string, sometimes a list
        category = rec.get('category', 'unknown')
        if isinstance(category, list) and category:
            category = category[0]
            
        # Create standardized recommendation
        standard_rec = {
            'place_id': place_id,
            'name': rec.get('name', rec.get('row', {}).get('name', 'Unknown')),
            'score': rec.get('score', 0.0),
            'category': category,
            'rating': rec.get('rating', rec.get('row', {}).get('rating', 0.0)),
            'user_ratings_total': rec.get('user_ratings_total', rec.get('row', {}).get('user_ratings_total', 0)),
            'distance': rec.get('distance', 0.0),
            'lat': rec.get('lat', rec.get('row', {}).get('lat', 0.0)),
            'lng': rec.get('lng', rec.get('row', {}).get('lng', 0.0)),
            'vicinity': rec.get('vicinity', rec.get('row', {}).get('vicinity', '')),
            'types': rec.get('types', rec.get('row', {}).get('types', '')),
            'description': rec.get('description', rec.get('row', {}).get('description', '')),
            'icon': rec.get('icon', rec.get('row', {}).get('icon', 'https://via.placeholder.com/80')),
            'source': source
        }
        
        return standard_rec
    
    def get_recommendations(self, user_lat, user_lon, user_prefs, predicted_ratings_dict, num_recs=10):
        """
        Get ensemble recommendations by combining multiple models.
        
        Args:
            user_lat: User's latitude
            user_lon: User's longitude
            user_prefs: Dictionary of user category preferences
            predicted_ratings_dict: Dictionary of predicted category ratings
            num_recs: Number of recommendations to return
            
        Returns:
            list: Ensemble recommendations
        """
        # Reset category counter for each recommendation request
        self.recommended_categories = defaultdict(int)
        all_recommendations = []
        
        # Store a reference to the places DataFrame for all recommenders to use
        places_df = self.autoencoder_recommender.places_df if self.autoencoder_recommender else None
        
        # Get recommendations from each model
        if self.autoencoder_recommender:
            # Prepare input for autoencoder (needs array and mask)
            categories = list(predicted_ratings_dict.keys())
            user_prefs_array = np.array([predicted_ratings_dict.get(cat, 0) for cat in categories])
            provided_mask = np.array([cat in user_prefs and user_prefs[cat] > 0 for cat in categories])
            
            auto_recs = self.autoencoder_recommender.get_recommendations(
                user_lat=user_lat,
                user_lon=user_lon,
                user_prefs=user_prefs_array,
                provided_mask=provided_mask,
                num_recs=num_recs
            )
            auto_recs = self._normalize_scores(auto_recs)
            all_recommendations.extend([
                self._standardize_recommendation(rec, 'autoencoder') 
                for rec in auto_recs
            ])
        
        if self.svd_recommender and places_df is not None:
            svd_recs = self.svd_recommender.get_recommendations(
                df=places_df,  # Use the shared places_df instead
                user_lat=user_lat,
                user_lon=user_lon,
                predicted_ratings=predicted_ratings_dict,
                top_n=num_recs,
                max_distance=5  # Allow recommendations up to 5km away
            )
            svd_recs = self._normalize_scores(svd_recs)
            all_recommendations.extend([
                self._standardize_recommendation(rec, 'svd') 
                for rec in svd_recs
            ])
        
        if self.transfer_recommender and places_df is not None:
            transfer_recs = self.transfer_recommender.get_recommendations(
                user_preferences=user_prefs,
                user_lat=user_lat,
                user_lon=user_lon,
                places_df=places_df,  # Use the shared places_df
                top_n=num_recs
            )
            transfer_recs = self._normalize_scores(transfer_recs)
            all_recommendations.extend([
                self._standardize_recommendation(rec, 'transfer') 
                for rec in transfer_recs
            ])
            
        if self.madrid_transfer_recommender:
            madrid_recs = self.madrid_transfer_recommender.get_recommendations(
                user_lat=user_lat,
                user_lon=user_lon,
                user_prefs=user_prefs,
                num_recs=num_recs
            )
            madrid_recs = self._normalize_scores(madrid_recs)
            all_recommendations.extend([
                self._standardize_recommendation(rec, 'madrid_transfer') 
                for rec in madrid_recs
            ])
            
        # Apply weights based on source
        for rec in all_recommendations:
            source = rec.get('source', '')
            weight = self.weights.get(source, 1.0)
            rec['score'] *= weight
            
        # Apply adjustments for better recommendations
        all_recommendations = self._apply_distance_penalty(all_recommendations, max_distance=5.0)
        all_recommendations = self._apply_diversity_boost(all_recommendations, max_per_category=2)
        all_recommendations = self._apply_novelty_boost(all_recommendations)
        
        # Remove duplicates by place_id
        unique_places = {}
        for rec in all_recommendations:
            place_id = rec.get('place_id', '')
            if place_id and (place_id not in unique_places or rec['score'] > unique_places[place_id]['score']):
                unique_places[place_id] = rec
                
        # Sort by final score and return top recommendations
        final_recommendations = sorted(unique_places.values(), key=lambda x: x['score'], reverse=True)
        
        # Return only requested number of recommendations
        return final_recommendations[:num_recs]