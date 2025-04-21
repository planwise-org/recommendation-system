import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import math
from collections import defaultdict
from .base_recommender import BaseRecommender

class MadridTransferRecommender(BaseRecommender):
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2', embedding_path='models/madrid_place_embeddings.npz'):
        self.places_df = pd.read_csv("resources/combined_places.csv")
        self.places_df['types_processed'] = self.places_df['types'].fillna('').apply(
            lambda x: [t.strip().lower() for t in x.split(',')]
        )

        data = np.load(embedding_path, allow_pickle=True)
        self.embeddings = data['embeddings']
        self.place_ids = data['place_id']

        self.embedding_model = SentenceTransformer(embedding_model_name)

    def _preferences_to_embedding(self, preferences):
        pref_text = ' '.join([cat for cat, rating in preferences.items() for _ in range(int(rating))])
        return self.embedding_model.encode(pref_text)

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lambda = math.radians(lon2 - lon1)
        a = math.sin(d_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def get_recommendations(self, user_lat, user_lon, user_prefs, num_recs=10):
        user_emb = self._preferences_to_embedding(user_prefs)
        similarities = cosine_similarity([user_emb], self.embeddings)[0]

        recs = []
        for idx, place_id in enumerate(self.place_ids):
            row = self.places_df[self.places_df['place_id'] == place_id].iloc[0]
            lat, lon = row.get('lat'), row.get('lng')
            distance = self._haversine_distance(user_lat, user_lon, lat, lon)
            distance_km = distance / 1000

            if distance_km > 3:
                continue

            best_cat = row['types_processed'][0] if row['types_processed'] else 'other'

            recs.append({
                "place_id": place_id,
                "name": row["name"],
                "rating": row.get('rating', 0.0),
                "user_ratings_total": row.get('user_ratings_total', 0),
                "types": row["types"],
                "types_processed": row["types_processed"],
                "category": best_cat,
                "lat": lat,
                "lng": lon,
                "vicinity": row.get("vicinity", ""),
                "description": row.get("description", ""),
                "distance": distance,
                "similarity": similarities[idx],
                "icon": row.get("icon", "https://via.placeholder.com/80")
            })

        sorted_recs = sorted(recs, key=lambda x: x["similarity"], reverse=True)

        grouped = defaultdict(list)
        for rec in sorted_recs:
            if len(grouped[rec['category']]) < 2:
                grouped[rec['category']].append(rec)

        final_recs = []
        while len(final_recs) < num_recs:
            added = False
            for cat, items in grouped.items():
                if items:
                    final_recs.append(items.pop(0))
                    added = True
                    if len(final_recs) == num_recs:
                        break
            if not added:
                break

        return final_recs
