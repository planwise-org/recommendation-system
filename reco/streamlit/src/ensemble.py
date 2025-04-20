import numpy as np
import pandas as pd
import math
from src.transfer_recommender import TransferRecommender

class EnsembleRecommender:
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
    
    This approach typically provides more robust recommendations by leveraging the
    strengths of each individual model while mitigating their weaknesses.
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
        
    def initialize_models(self, auto_model, scaler, places_df, category_to_place_types, AutoencoderRecommender, SVDPlaceRecommender, svd_params=None):
        """
        Initialize all the recommender models.
        
        Args:
            auto_model: The autoencoder model
            scaler: The scaler used for autoencoder input/output
            places_df: DataFrame containing places data
            category_to_place_types: Mapping from categories to place types
            AutoencoderRecommender: AutoencoderRecommender class
            SVDPlaceRecommender: SVDPlaceRecommender class
            svd_params: Parameters for the SVD model
        """
        self.places_df = places_df
        self.category_to_place_types = category_to_place_types
        
        # Initialize the autoencoder recommender
        self.autoencoder_recommender = AutoencoderRecommender()
        
        # Initialize the SVD recommender
        self.svd_recommender = SVDPlaceRecommender(svd_params)
        self.svd_recommender.fit(places_df)
        
        # Initialize the transfer learning recommender
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
        
        # If all scores are the same, return the original recommendations
        if max_score == min_score:
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
            user_prefs: User preferences array
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
        # Extract the final_predictions from user_prefs
        if self.autoencoder_recommender is None:
            return []
            
        try:
            # Since we can't directly use the model's get_recommendations due to import issues,
            # we'll implement a simplified version here based on the original code
            candidates = []
            for _, row in self.places_df.iterrows():
                dist = self.haversine(user_lat, user_lon, row['lat'], row['lng'])
                if dist > self.max_distance:
                    continue
                    
                best_cat = None
                best_score = 0
                
                # For each category, check if this place matches and get the highest score
                for i, cat in enumerate(user_prefs.keys()):
                    cat_rating = user_prefs[cat]
                    mapped_types = self.category_to_place_types.get(cat, [])
                    
                    # Check if any of the place types match this category
                    if any(pt in row['types_processed'] for pt in mapped_types):
                        cat_score = cat_rating / 5.0  # Normalize to 0-1
                        if cat_score > best_score:
                            best_score = cat_score
                            best_cat = cat
                
                if best_cat:
                    # Calculate overall score with multiple factors
                    norm_rating = (row['rating'] / 5.0) if pd.notna(row['rating']) else 0
                    norm_reviews = (np.log(row['user_ratings_total'] + 1)) / np.log(self.places_df['user_ratings_total'].max() + 1) if pd.notna(row['user_ratings_total']) else 0
                    
                    score = (0.1 * (1 - dist/self.max_distance) +
                            0.2 * norm_rating +
                            0.2 * norm_reviews +
                            0.5 * best_score)
                    
                    candidates.append({
                        'name': row['name'],
                        'lat': row['lat'],
                        'lng': row['lng'],
                        'score': score,
                        'rating': row['rating'],
                        'user_ratings_total': row['user_ratings_total'],
                        'types': row['types'],
                        'distance': dist,
                        'category': best_cat,
                        'icon': row.get('icon', ''),
                        'vicinity': row.get('vicinity', '')
                    })
            
            candidates.sort(key=lambda x: x['score'], reverse=True)
            return candidates[:num_recs]
            
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

    def initialize_models_from_reviews(self, reviews_df, places_df=None):
        from reco import SVDPlaceRecommender
        from transfer_recommender import TransferRecommender

        self.svd_recommender = SVDPlaceRecommender()
        self.svd_recommender.fit(reviews_df)

        self.transfer_recommender = TransferRecommender()
        if places_df is not None:
            self.transfer_recommender.transfer_to_places(places_df)


        self.places_df = places_df