import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import math
from collections import defaultdict
from .base_recommender import BaseRecommender

class MadridTransferRecommender(BaseRecommender):
    def __init__(self,
                 embedding_model_name='all-MiniLM-L6-v2',
                 madrid_emb_path='models/madrid_place_embeddings.npz',
                 ca_user_emb_path='models/user_embeddings.npz'):
        # Load Madrid place metadata & embeddings
        self.places_df = pd.read_csv("resources/combined_places.csv")
        self.places_df['types_processed'] = (
            self.places_df['types']
              .fillna('')
              .apply(lambda x: [t.strip().lower() for t in x.split(',')])
        )
        data = np.load(madrid_emb_path, allow_pickle=True)
        self.madrid_embeddings = data['embeddings']
        self.madrid_place_ids   = data['place_id']

        # Load California user embeddings
        ca_data = np.load(ca_user_emb_path, allow_pickle=True)
        self.ca_user_embeddings = ca_data['embeddings']  # shape (n_users, dim)
        self.ca_user_ids        = ca_data['user_ids']    # shape (n_users,)

        # SentenceTransformer for embedding new user preferences
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def _preferences_to_embedding(self, preferences):
        # Turn category ratings into a weighted "pseudo‐document"
        pref_text = ' '.join(
            [cat for cat, rating in preferences.items() for _ in range(int(rating))]
        )
        return self.embedding_model.encode(pref_text)

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371000
        φ1, φ2 = math.radians(lat1), math.radians(lat2)
        dφ = math.radians(lat2 - lat1)
        dλ = math.radians(lon2 - lon1)
        a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    def get_recommendations(self, user_lat, user_lon, user_prefs, num_recs=10):
        # 1. Embed the new Madrid user's category preferences
        pref_emb = self._preferences_to_embedding(user_prefs)

        # 2. Find most similar California user
        sims_to_ca = cosine_similarity([pref_emb], self.ca_user_embeddings)[0]
        best_idx = np.argmax(sims_to_ca)
        ca_user_emb = self.ca_user_embeddings[best_idx]

        # 3. Compute similarity vs. Madrid places using that CA user embedding
        sims_to_madrid = cosine_similarity([ca_user_emb], self.madrid_embeddings)[0]

        # 4. Gather and filter by distance
        candidates = []
        for idx, place_id in enumerate(self.madrid_place_ids):
            row = self.places_df[self.places_df['place_id'] == place_id].iloc[0]
            dist = self._haversine_distance(user_lat, user_lon, row['lat'], row['lng']) 
            if dist > 2000:  # optional max radius in km
                continue
            candidates.append({
                'place_id': place_id,
                'name': row['name'],
                'distance': dist,
                'similarity': sims_to_madrid[idx],
                'score': sims_to_madrid[idx],  
                'lat': row['lat'],
                'lng': row['lng'],
                'category': row['types_processed'][0] if row['types_processed'] and isinstance(row['types_processed'], list) else 'other',
                'actual_rating': row['rating'],
                'user_ratings_total': row['user_ratings_total'],
                'description': row.get('description', ''),
                'types': row['types'],
                'vicinity': row.get('vicinity',''),
                'icon': row.get('icon',''),
            })

        # 5. Rank & return top‐N
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:num_recs]
