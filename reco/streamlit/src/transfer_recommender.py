import pandas as pd
import numpy as np
from recommenders.datasets import movielens
from recommenders.models.sar import SAR
from recommenders.datasets.python_splitters import python_random_split
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

class TransferRecommender:
    def __init__(self, embedding_dim=64):
        self.embedding_dim = embedding_dim
        self.model = None
        self.movie_embeddings = None
        self.movie_ids = None
        self.place_embeddings = None
        
        # Expanded and improved category to Google Places type mapping
        self.type_category_mapping = {
            # Nature related
            "park": ["nature", "parks"],
            "natural_feature": ["nature"],
            "campground": ["nature"],
            "national_park": ["nature"],
            "forest": ["nature"],
            "garden": ["nature", "gardens"],
            "beach": ["nature", "beaches"],
            "hiking_area": ["nature"],
            
            # Museums and culture
            "museum": ["museums"],
            "art_gallery": ["art galleries"],
            "tourist_attraction": ["tourist attractions"],
            
            # Food and drink
            "restaurant": ["restaurants"],
            "cafe": ["cafes"],
            "bar": ["pubs/bars"],
            "bakery": ["bakeries"],
            "supermarket": ["supermarket"],
            
            # Entertainment
            "amusement_park": ["amusement parks"],
            "movie_theater": ["theatres"],
            "night_club": ["dance clubs"],
            "shopping_mall": ["malls"],
            
            # Others
            "spa": ["beauty & spas"],
            "lodging": ["hotels/other lodgings"],
            "resort": ["resorts"],
            "swimming_pool": ["swimming pools"]
        }
        
        # Keep your existing genre mapping
        self.category_genre_mapping = {
            "resorts": ["Adventure", "Family"],
            "burger/pizza shops": ["Comedy"],
            "hotels/other lodgings": ["Drama"],
            "juice bars": ["Family"],
            "beauty & spas": ["Romance"],
            "gardens": ["Documentary", "Family"],
            "amusement parks": ["Action", "Adventure", "Family"],
            "farmer market": ["Documentary"],
            "market": ["Documentary"],
            "music halls": ["Musical"],
            "nature": ["Documentary", "Adventure"],
            "tourist attractions": ["Adventure", "Documentary"],
            "beaches": ["Adventure", "Romance"],
            "parks": ["Family", "Adventure"],
            "theatres": ["Drama"],
            "museums": ["Documentary", "History"],
            "malls": ["Comedy"],
            "restaurants": ["Comedy", "Romance"],
            "pubs/bars": ["Comedy"],
            "local services": ["Drama"],
            "art galleries": ["Documentary"],
            "dance clubs": ["Musical"],
            "swimming pools": ["Sport"],
            "bakeries": ["Family"],
            "cafes": ["Romance", "Comedy"],
            "view points": ["Adventure"],
            "monuments": ["History", "Documentary"],
            "zoo": ["Documentary", "Family"],
            "supermarket": ["Family"]
        }
        
    def train_base_model(self, save_path="models"):
        """Train on MovieLens data and save the model"""
        try:
            if os.path.exists(f"{save_path}/movie_embeddings.joblib"):
                print("Loading pre-trained movie embeddings...")
                # Delete the existing file if it's corrupted
                os.remove(f"{save_path}/movie_embeddings.joblib")
            
            print("Loading MovieLens data...")
            df = movielens.load_pandas_df(
                size='100k',
                header=['userId', 'movieId', 'rating', 'timestamp']
            )
            
            print("Preparing training data...")
            train, test = python_random_split(df, ratio=0.75)
            
            print("Training base model...")
            self.model = SAR(
                similarity_type="jaccard",
                time_decay_coefficient=30,
                timedecay_formula=True,
                col_user='userId',
                col_item='movieId',
                col_rating='rating',
                col_timestamp='timestamp'
            )
            
            self.model.fit(train)
            
            # Extract and save item similarity matrix as embeddings
            print("Extracting movie embeddings...")
            self.movie_embeddings = self.model.item_similarity
            self.movie_ids = np.array(sorted(df['movieId'].unique()))
            
            # Save embeddings and movie IDs
            os.makedirs(save_path, exist_ok=True)
            np.savez(
                f"{save_path}/movie_embeddings.npz",
                embeddings=self.movie_embeddings,
                movie_ids=self.movie_ids
            )
            print("Movie embeddings saved!")
            
        except Exception as e:
            print(f"Error in train_base_model: {str(e)}")
            # Initialize with empty embeddings if something goes wrong
            self.movie_embeddings = np.random.normal(0, 0.1, size=(100, self.embedding_dim))
            self.movie_ids = np.arange(100)

    def transfer_to_places(self, places_df, save_path="models"):
        """Transfer learned patterns to places domain"""
        if os.path.exists(f"{save_path}/place_embeddings.joblib"):
            print("Loading pre-computed place embeddings...")
            self.place_embeddings = joblib.load(f"{save_path}/place_embeddings.joblib")
            return

        print("Transferring knowledge to places domain...")
        self.place_embeddings = {}
        
        for idx, row in places_df.iterrows():
            if idx % 100 == 0:
                print(f"Processing place {idx}/{len(places_df)}")
                
            # Get place types and normalize them
            place_types = str(row['types']).split(',') if pd.notna(row['types']) else []
            place_types = [t.strip().lower() for t in place_types]
            
            # Match place types to categories using the mapping
            matched_categories = set()
            for place_type in place_types:
                if place_type in self.type_category_mapping:
                    matched_categories.update(self.type_category_mapping[place_type])
            
            # Collect relevant movie embeddings
            relevant_embeddings = []
            for category in matched_categories:
                if category in self.category_genre_mapping:
                    movie_indices = self._get_movies_by_genre(None)
                    try:
                        embeddings = [self.movie_embeddings[idx] for idx in movie_indices]
                        relevant_embeddings.extend(embeddings)
                    except IndexError:
                        continue
            
            # Create place embedding
            if relevant_embeddings:
                self.place_embeddings[str(row['place_id'])] = np.mean(relevant_embeddings, axis=0)
            else:
                self.place_embeddings[str(row['place_id'])] = np.random.normal(
                    0, 0.1, 
                    size=self.movie_embeddings.shape[1]
                )
        
        # Save place embeddings
        joblib.dump(self.place_embeddings, f"{save_path}/place_embeddings.joblib")
        print("Place embeddings saved!")

    def _get_movies_by_genre(self, genre):
        """Get random movie indices (not IDs) for similarity lookup"""
        n_movies = self.movie_embeddings.shape[0]  # Get actual size of embedding matrix
        return np.random.choice(range(n_movies), size=min(5, n_movies))  # Ensure we don't request more than we have

    def get_recommendations(self, user_preferences, user_lat, user_lon, places_df, top_n=10):
        """Get recommendations using transferred knowledge"""
        # Convert user preferences to embedding
        user_embedding = self._preferences_to_embedding(user_preferences)
        
        recommendations = []
        for idx, row in places_df.iterrows():
            place_id = str(row['place_id'])
            if place_id in self.place_embeddings:
                # Calculate similarity
                similarity = cosine_similarity(
                    [user_embedding],
                    [self.place_embeddings[place_id]]
                )[0][0]
                
                # Calculate distance score
                distance = self._haversine_distance(
                    user_lat, user_lon,
                    row['lat'], row['lng']
                )
                distance_score = 1 / (1 + distance * 0.1)
                
                # Get place types for debugging
                place_types = str(row['types']).split(',') if pd.notna(row['types']) else []
                place_types = [t.strip().lower() for t in place_types]
                
                # Strong preference boost for exact type matches
                preference_boost = 0
                for place_type in place_types:
                    if place_type in self.type_category_mapping:
                        matched_categories = self.type_category_mapping[place_type]
                        for category, rating in user_preferences.items():
                            if rating > 0 and category.lower() in matched_categories:
                                # Much stronger boost for exact matches
                                preference_boost += (rating / 5.0) * 2.0  # Doubled the boost

                # Heavily weight exact matches in final score
                final_score = (0.3 * similarity + 0.5 * preference_boost + 0.2 * distance_score)
                
                # Only add places that have some relevance to preferences
                if preference_boost > 0 or similarity > 0.5:
                    recommendations.append({
                        'place_id': place_id,
                        'name': row['name'],
                        'score': final_score,
                        'distance': distance,
                        'similarity': similarity,
                        'preference_boost': preference_boost,
                        'types': row['types'] if 'types' in row else None,
                        'rating': row['rating'] if 'rating' in row else None,
                        'lat': row['lat'],
                        'lng': row['lng']
                    })
        
        # Sort by score and print top recommendations for debugging
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        print("\nTop recommendations with types:")
        for r in recommendations[:min(3, len(recommendations))]:
            print(f"{r['name']} (Score: {r['score']:.2f}, Types: {r['types']}, Distance: {r['distance']:.2f}km)")
        
        return recommendations[:top_n]

    def _preferences_to_embedding(self, preferences):
        """Convert user category preferences to embedding space"""
        relevant_embeddings = []
        
        for category, rating in preferences.items():
            if category in self.category_genre_mapping:
                # Normalize rating to 0-1
                weight = rating / 5.0
                
                # Get relevant genre embeddings
                for genre in self.category_genre_mapping[category]:
                    movie_indices = self._get_movies_by_genre(genre)
                    if len(movie_indices) > 0:  # Changed from if genre_movies to check length
                        genre_embedding = np.mean([self.movie_embeddings[idx] for idx in movie_indices], axis=0)
                        relevant_embeddings.append(weight * genre_embedding)
        
        if relevant_embeddings:
            return np.mean(relevant_embeddings, axis=0)
        return np.random.normal(0, 0.1, size=self.movie_embeddings.shape[1])

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate the great circle distance between two points"""
        R = 6371  # Earth's radius in kilometers

        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c