import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class MadridTransferRecommender:
    def __init__(self, embedding_model=None):
        # Load Madrid metadata
        self.places_df = pd.read_csv("resources/combined_places.csv")
        self.places_df['types_processed'] = self.places_df['types'].fillna('').apply(
            lambda x: [t.strip().lower() for t in x.split(',')]
        )

        # Load Madrid embeddings
        data = np.load("models/madrid_place_embeddings.npz", allow_pickle=True)
        self.embeddings = data['embeddings']
        self.place_ids = data['place_id']

        # Embedding model
        self.embedding_model = embedding_model or SentenceTransformer('all-MiniLM-L6-v2')

    def _preferences_to_embedding(self, preferences):
        # Generate weighted string from categories
        pref_text = ' '.join([cat for cat, rating in preferences.items() for _ in range(int(rating))])
        return self.embedding_model.encode(pref_text)

    def get_recommendations(self, user_preferences, user_lat, user_lon, top_n=10):
        user_emb = self._preferences_to_embedding(user_preferences)
        sims = cosine_similarity([user_emb], self.embeddings)[0]

        ranked_indices = sims.argsort()[::-1][:top_n]
        recs = []
        for idx in ranked_indices:
            row = self.places_df[self.places_df['place_id'] == self.place_ids[idx]].iloc[0]
            recs.append({
                "place_id": row['place_id'],
                "name": row['name'],
                "rating": row['rating'],
                "user_ratings_total": row['user_ratings_total'],
                "types": row['types'],
                "lat": row['lat'],
                "lng": row['lng'],
                "vicinity": row.get('vicinity', ''),
                "description": row.get('description', ''),
                "score": sims[idx],
                "icon": row.get('icon', 'https://via.placeholder.com/80')
            })
        return recs