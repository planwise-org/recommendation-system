"""
Transfer learning based recommendation model.
"""

import pandas as pd
import numpy as np
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_random_split
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import torch
import cornac
from recommenders.models.cornac.cornac_utils import predict_ranking
from recommenders.utils.constants import SEED
from .base_recommender import BaseRecommender

class TransferRecommender(BaseRecommender):
    """
    A recommender that uses transfer learning from MovieLens data to 
    recommend places. It leverages knowledge learned from the movie 
    domain to make recommendations in the place domain.
    """
    
    def __init__(self, embedding_dim=64):
        """
        Initialize the transfer recommender.
        
        Args:
            embedding_dim (int): Dimension of embedding vectors
        """
        self.embedding_dim = embedding_dim
        self.model = None
        self.movie_embeddings = None
        self.movie_ids = None
        self.place_embeddings = None
        
        # BiVAE parameters
        self.latent_dim = embedding_dim
        self.encoder_dims = [100]
        self.act_func = "tanh"
        self.likelihood = "pois"
        self.num_epochs = 200
        self.batch_size = 128
        self.learning_rate = 0.001
        
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
        """Train on MovieLens data using BiVAE and save the model"""
        try:
            # Check if pre-computed embeddings exist and load them
            if os.path.exists(f"{save_path}/movie_embeddings.npz"):
                print("Loading pre-trained movie embeddings...")
                data = np.load(f"{save_path}/movie_embeddings.npz", allow_pickle=True)
                self.movie_embeddings = data['embeddings']
                self.movie_ids = data['movie_ids']
                return
                
            print("Loading MovieLens data...")
            df = movielens.load_pandas_df(
                size='100k',
                header=['userId', 'movieId', 'rating', 'timestamp']
            )
            
            # For BiVAE, we're only interested in userId, movieId and rating
            df = df[['userId', 'movieId', 'rating']]
            
            print("Preparing training data...")
            train, test = python_random_split(df, ratio=0.75)
            
            # Create id mappings to preserve original movieIds
            print("Creating ID mappings...")
            user_ids = train['userId'].unique()
            movie_ids = train['movieId'].unique()
            
            user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
            movie_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
            idx_to_movie = {i: mid for mid, i in movie_to_idx.items()}
            
            # Convert train data to use indices
            train_cornac = train.copy()
            train_cornac['userId'] = train_cornac['userId'].map(user_to_idx)
            train_cornac['movieId'] = train_cornac['movieId'].map(movie_to_idx)
            
            print("Creating Cornac dataset...")
            train_set = cornac.data.Dataset.from_uir(
                train_cornac.rename(columns={'userId': 'userID', 'movieId': 'itemID'})
                .itertuples(index=False), 
                seed=SEED
            )
            
            print(f"Number of users: {train_set.num_users}")
            print(f"Number of items: {train_set.num_items}")
            
            print("Training BiVAE model...")
            self.model = cornac.models.BiVAECF(
                k=self.latent_dim,
                encoder_structure=self.encoder_dims,
                act_fn=self.act_func,
                likelihood=self.likelihood,
                n_epochs=self.num_epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                seed=SEED,
                use_gpu=torch.cuda.is_available(),
                verbose=True
            )
            
            self.model.fit(train_set)
            
            # Extract and save item embeddings from BiVAE
            print("Extracting movie embeddings...")
            # In BiVAE, we need to extract embeddings from the model directly
            # Get item embeddings using the feature matrices of BiVAE
            # This is the correct way to access embeddings in BiVAECF
            self.movie_embeddings = self.model.beta_item if hasattr(self.model, 'beta_item') else None
            
            if self.movie_embeddings is None:
                # Fallback approach: generate embeddings by getting latent factors
                item_ids = np.arange(train_set.num_items)
                self.movie_embeddings = self.model.get_item_vectors()
            
            # Get original movie IDs in the same order as the embeddings
            self.movie_ids = np.array([idx_to_movie[i] for i in range(len(movie_ids))])
            
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

        if self.movie_embeddings is None:
            print("Movie embeddings not available. Loading or training base model...")
            self.train_base_model(save_path)

        print("Transferring knowledge to places domain...")
        self.place_embeddings = {}
        
        # Get embedding dimension from actual embeddings
        embedding_dim = self.movie_embeddings.shape[1] if hasattr(self.movie_embeddings, 'shape') else self.embedding_dim
        
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
                # For places without matching categories, create random embeddings with same shape
                self.place_embeddings[str(row['place_id'])] = np.random.normal(0, 0.1, size=embedding_dim)
        
        # Save place embeddings
        joblib.dump(self.place_embeddings, f"{save_path}/place_embeddings.joblib")
        print("Place embeddings saved!")

    def _get_movies_by_genre(self, genre):
        """Get random movie indices (not IDs) for similarity lookup"""
        n_movies = self.movie_embeddings.shape[0]  # Get actual size of embedding matrix
        return np.random.choice(range(n_movies), size=min(5, n_movies))  # Ensure we don't request more than we have

    def get_recommendations(self, user_preferences, user_lat, user_lon, places_df, top_n=10):
        """
        Get recommendations using transferred knowledge
        
        Args:
            user_preferences (dict): User category preferences
            user_lat (float): User's latitude
            user_lon (float): User's longitude 
            places_df (DataFrame): Places data
            top_n (int): Number of recommendations to return
            
        Returns:
            list: Recommended places
        """
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
                        'icon': row.get('icon', ''),
                        'user_ratings_total': row.get('user_ratings_total', 0),
                        'description': row.get('description', ''),
                        'score': final_score,
                        'distance': distance,
                        'similarity': similarity,
                        'preference_boost': preference_boost,
                        'types': row.get('types', ''),
                        'rating': row.get('rating', 0),
                        'lat': row['lat'],
                        'lng': row['lng'],
                        'vicinity': row.get('vicinity', '')
                    })
        
        # Sort by score and print top recommendations for debugging
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
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