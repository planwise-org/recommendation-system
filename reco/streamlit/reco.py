import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
from surprise import accuracy
from collections import defaultdict
from math import radians, sin, cos, sqrt, atan2
import os

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

        self.category_to_place_types = {
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

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate the great circle distance between two points"""
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        r = 6371
        return c * r

    def get_type_score(self, place_types, predicted_ratings):
        """Calculate score based on place types and user preferences"""
        if pd.isna(place_types):
            return 0

        types_list = str(place_types).split(', ')
        scores = []

        # Convert category preferences to type preferences
        type_preferences = {}
        for category, rating in predicted_ratings.items():
            if category in self.category_to_place_types:
                for place_type in self.category_to_place_types[category]:
                    if place_type in type_preferences:
                        type_preferences[place_type] = max(type_preferences[place_type], rating)
                    else:
                        type_preferences[place_type] = rating

        # Calculate score for each matching type
        for place_type in types_list:
            if place_type in type_preferences:
                scores.append(type_preferences[place_type])

        # Return average score if any matches found
        return np.mean(scores) if scores else 0

    def prepare_data(self, df):
        """Prepare data for SVD"""
        reader = Reader(rating_scale=(1, 5))

        ratings_dict = {
            'user': [],
            'item': [],
            'rating': []
        }

        for idx, row in df.iterrows():
            ratings_dict['item'].append(row['place_id'])
            ratings_dict['user'].append(f'user_{idx % 100}')
            ratings_dict['rating'].append(row['rating'] if pd.notna(row['rating']) else 3.0)

        ratings_df = pd.DataFrame(ratings_dict)
        return Dataset.load_from_df(ratings_df[['user', 'item', 'rating']], reader)

    def evaluate_model(self, df):
        """Evaluate the SVD model using cross-validation"""
        data = self.prepare_data(df)

        print("\nPerforming 5-fold cross-validation...")
        cv_results = cross_validate(self.model, data, measures=['RMSE', 'MAE'],
                                  cv=5, verbose=False)

        print("\nSVD Model Evaluation Metrics:")
        print("=" * 40)
        print(f"RMSE: {cv_results['test_rmse'].mean():.4f} (+/- {cv_results['test_rmse'].std():.4f})")
        print(f"MAE:  {cv_results['test_mae'].mean():.4f} (+/- {cv_results['test_mae'].std():.4f})")
        print("=" * 40)

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

    def get_recommendations(self, df, user_lat, user_lon, predicted_ratings, top_n=10, max_distance=None):
        """Get recommendations considering SVD, distance, and category preferences"""
        predictions = []

        for idx, row in df.iterrows():
            if pd.isna(row['rating']):
                continue


            distance = self.haversine_distance(user_lat, user_lon, row['lat'], row['lng'])
            if max_distance and distance > max_distance:
                continue

            svd_pred = self.model.predict('user_0', row['place_id']).est
            type_score = self.get_type_score(row['types'], predicted_ratings)

            # Combined score (weighted average of SVD, type preference, and distance)
            distance_score = 1 / (1 + distance * 0.1)
            score = (
                0.4 * svd_pred/5.0 +
                0.4 * type_score/5.0 +
                0.2 * distance_score
            )

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
                'score': score
            })

        recommendations = sorted(predictions, key=lambda x: x['score'], reverse=True)[:top_n]

        print("\nTop Recommended Places:")
        print(f"{'Name':<30} {'Distance (km)':<15} {'Type Score':<12} {'Rating':<8} {'Reviews':<8}")
        print("-" * 80)

        for rec in recommendations:
            name = rec['name'][:27] + "..." if len(rec['name']) > 27 else rec['name']
            print(f"{name:<30} {rec['distance']:<15.2f} {rec['type_score']:<12.1f} {rec['actual_rating']:<8.1f} {int(rec['user_ratings_total']):<8}")

        print("\nDetailed Information:")
        print("=" * 80)

        for rec in recommendations:
            print(f"\nName: {rec['name']}")
            print(f"Distance from you: {rec['distance']:.2f} km")
            print(f"Address: {rec['vicinity']}")
            print(f"Type Preference Score: {rec['type_score']:.1f}")
            print(f"Rating: {rec['actual_rating']} ({int(rec['user_ratings_total'])} reviews)")
            print(f"Types: {rec['types']}")
            print("-" * 80)

        return recommendations

def main():
    # Example predicted ratings from nn
    predicted_ratings = {
        "resorts": 1.5,
        "burger/pizza shops": 1.0,
        "hotels/other lodgings": 2.8,
        "juice bars": 3.9,
        "beauty & spas": 3.2,
        "gardens": 3.0,
        "amusement parks": 2.8,
        "farmer market": 3.7,
        "market": 4.1,
        "music halls": 3.2,
        "nature": 3.6,
        "tourist attractions": 4.5,
        "beaches": 3.8,
        "parks": 3.9,
        "theatres": 3.0,
        "museums": 4.8,
        "malls": 3.6,
        "restaurants": 1.7,
        "pubs/bars": 4.0,
        "local services": 3.0,
        "art galleries": 4.7,
        "dance clubs": 3.8,
        "swimming pools": 3.5,
        "bakeries": 4.0,
        "cafes": 1.2,
        "view points": 4.0,
        "monuments": 4.9,
        "zoo": 4.5,
        "supermarket": 0.0
    }
    BASE_PATH = "reco/streamlit/"
    if os.environ.get("ENV") == "prod":
        df = pd.read_csv(BASE_PATH + 'combined_places.csv')
    else:
        df = pd.read_csv('combined_places.csv')

    recommender = SVDPlaceRecommender()

    evaluation_metrics = recommender.evaluate_model(df)

    recommender.fit(df)

    #  user location
    user_lat = 40.3750556
    user_lon = -3.7747428

    recommendations = recommender.get_recommendations(
        df,
        user_lat,
        user_lon,
        predicted_ratings,
        top_n=10,
        max_distance=5  # within 5km
    )

if __name__ == "__main__":
    main()
