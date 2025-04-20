"""
SVD-based recommendation model using collaborative filtering.
"""

import pandas as pd
import numpy as np
import math
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
from .base_recommender import BaseRecommender

class SVDPlaceRecommender(BaseRecommender):
    """
    A recommender that uses Singular Value Decomposition (SVD) for collaborative filtering
    to recommend places based on user preferences and location.
    """
    
    def __init__(self, svd_params=None, category_to_place_types=None):
        """
        Initialize the SVD recommender.
        
        Args:
            svd_params: SVD parameters for the model
            category_to_place_types: Dictionary mapping categories to place types
        """
        self.svd_params = svd_params or {
            "n_factors": 150,
            "n_epochs": 10,
            "lr_all": 0.002,
            "reg_all": 0.02,
            "random_state": 42,
            "verbose": False
        }
        self.model = SVD(**self.svd_params)
        self.category_to_place_types = category_to_place_types if category_to_place_types is not None else {}

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate the great circle distance (in km) between two points"""
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        r = 6371
        return c * r

    def get_type_score(self, place_types, predicted_ratings):
        """Calculate score based on place types and user preferences"""
        if pd.isna(place_types):
            return 0
            
        if self.category_to_place_types is None or not self.category_to_place_types:
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
        """Prepare data for SVD"""
        reader = Reader(rating_scale=(1, 5))
        ratings_dict = {'user': [], 'item': [], 'rating': []}
        
        for idx, row in df.iterrows():
            ratings_dict['item'].append(row['place_id'])
            ratings_dict['user'].append(f'user_{idx % 100}')
            ratings_dict['rating'].append(row['rating'] if pd.notna(row['rating']) else 3.0)
            
        ratings_df = pd.DataFrame(ratings_dict)
        return Dataset.load_from_df(ratings_df[['user', 'item', 'rating']], reader)

    def evaluate_model(self, df):
        """Evaluate the SVD model using cross-validation"""
        data = self.prepare_data(df)
        cv_results = cross_validate(self.model, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
        
        return {
            'rmse_mean': cv_results['test_rmse'].mean(),
            'rmse_std': cv_results['test_rmse'].std(),
            'mae_mean': cv_results['test_mae'].mean(),
            'mae_std': cv_results['test_mae'].std()
        }

    def fit(self, df):
        """Train the SVD model"""
        data = self.prepare_data(df)
        trainset = data.build_full_trainset()
        self.model.fit(trainset)

    def get_recommendations(self, df, user_lat, user_lon, predicted_ratings, top_n=10, max_distance=5):
        """
        Get recommendations considering SVD, distance, and category preferences
        
        Args:
            df (DataFrame): Places data
            user_lat (float): User's latitude
            user_lon (float): User's longitude
            predicted_ratings (dict): Dictionary of predicted category ratings
            top_n (int): Number of recommendations to return
            max_distance (float): Maximum distance in kilometers
            
        Returns:
            list: Recommended places
        """
        predictions = []

        for idx, row in df.iterrows():
            if pd.isna(row['rating']):
                continue
                
            distance = self.haversine_distance(user_lat, user_lon, row['lat'], row['lng'])
            if max_distance and distance > max_distance:
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
                'vicinity': row.get('vicinity', ''),
                'icon': row.get('icon', ''),
                'description': row.get('description', ''),
                'score': score,
                'lat': row['lat'],
                'lng': row['lng']
            })

        return sorted(predictions, key=lambda x: x['score'], reverse=True)[:top_n] 