import numpy as np
import pandas as pd
from collections import defaultdict
import requests
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from .base_recommender import BaseRecommender
from .autoencoder_recommender import AutoencoderRecommender
from .svd_recommender import SVDPlaceRecommender
from .transfer_recommender import TransferRecommender
from .madrid_transfer_recommender import MadridTransferRecommender

class MetaEnsembleRecommender(BaseRecommender):
    """
    A hybrid ensemble recommender that merges outputs from multiple models
    to deliver accurate, diverse, and robust place suggestions.
    Features:
    - Uses a metalearner to optimally combine model predictions
    - Normalizes and weights scores from each model
    - Boosts diversity, novelty; applies distance penalties
    - Deduplicates overlapping recommendations
    - Filters out poorly rated places
    - Modular helper methods for clarity and maintainability
    """

    def __init__(self, weights=None):
        self.weights = weights or {
            'autoencoder': 0.10,
            'svd': 0.10,
            'transfer': 0.40,
            'madrid_transfer': 0.40
        }
        self.autoencoder_recommender = None
        self.svd_recommender = None
        self.transfer_recommender = None
        self.madrid_transfer_recommender = None
        self.recommended_categories = defaultdict(int)
        
        # Metalearner components
        self.metalearner = LinearRegression()
        self.scaler = StandardScaler()
        self.model_features = ['autoencoder', 'svd', 'transfer', 'madrid_transfer']
        self.training_data = []
        self.is_metalearner_trained = False

    def initialize_models(self, auto_model, scaler, places_df, category_to_place_types):
        self.autoencoder_recommender = AutoencoderRecommender(
            auto_model=auto_model,
            scaler=scaler,
            places_df=places_df,
            categories_list=list(category_to_place_types.keys()),
            category_mappings=category_to_place_types
        )
        self.svd_recommender = SVDPlaceRecommender(category_to_place_types=category_to_place_types)
        self.svd_recommender.fit(places_df)
        self.transfer_recommender = TransferRecommender()
        self.transfer_recommender.train_base_model()
        self.transfer_recommender.transfer_to_places(places_df)
        self.madrid_transfer_recommender = MadridTransferRecommender(
            embedding_model_name='all-MiniLM-L6-v2',
            embedding_path='models/madrid_place_embeddings.npz'
        )

    def _normalize_scores(self, recs, min_score=0.0, max_score=1.0):
        if not recs:
            return []
        # Ensure key 'score'
        if 'score' not in recs[0]:
            for alt in ['similarity','predicted_rating','distance_score']:
                if alt in recs[0]:
                    for r in recs:
                        r['score'] = r[alt]
                    break
            else:
                for r in recs:
                    r['score'] = 0.5
                return recs
        scores = [r['score'] for r in recs]
        lo, hi = min(scores), max(scores)
        rng = hi - lo
        if rng == 0:
            norm = [0.5]*len(scores)
        else:
            norm = [((s-lo)/rng)*(max_score-min_score)+min_score for s in scores]
        for i,r in enumerate(recs):
            r['original_score'] = r['score']
            r['score'] = norm[i]
        return recs

    def _apply_distance_penalty(self, recs, max_distance=5.0):
        for r in recs:
            d = r.get('distance',0)
            km = d/1000 if d>100 else d
            if km<=2.0:
                factor = 1.0 - (km*0.05)
            elif km<=max_distance:
                factor = 0.9 - ((km-2.0)*0.1)
            else:
                factor = 0.6 - ((km-max_distance)*0.2)
            factor = max(0.1, factor)
            r['distance_factor'] = factor
            r['score'] *= factor
        return recs

    def _apply_diversity_boost(self, recs, max_per_category=2):
        counts = defaultdict(int)
        for r in recs:
            cat = r.get('category','unknown')
            counts[cat]+=1
            if counts[cat] <= max_per_category:
                r['score'] *= (1.1 - 0.1*(counts[cat]-1))
            else:
                r['score'] *= 0.7
        return recs

    def _apply_novelty_boost(self, recs):
        for r in recs:
            reviews = r.get('user_ratings_total',0) or 0
            if reviews<10:
                boost=1.25
            elif reviews<50:
                boost=1.15
            elif reviews<100:
                boost=1.05
            else:
                boost=1.0
            r['novelty_boost'] = boost
            r['score'] *= boost
        return recs

    def _standardize(self, rec, source):
        place_id = rec.get('place_id','')
        cat = rec.get('category','unknown')
        if isinstance(cat,list):
            cat = cat[0] if cat else 'unknown'
        return {
            'place_id': place_id,
            'name': rec.get('name',rec.get('row',{}).get('name','Unknown')),
            'score': rec.get('score',0.0),
            'category': cat,
            'icon': rec.get('icon', 'https://via.placeholder.com/80'),
            'rating': rec.get('rating',rec.get('row',{}).get('rating',0.0)),
            'user_ratings_total': rec.get('user_ratings_total',rec.get('row',{}).get('user_ratings_total',0)),
            'distance': rec.get('distance',0.0),
            'lat': rec.get('lat',rec.get('row',{}).get('lat',0.0)),
            'lng': rec.get('lng',rec.get('row',{}).get('lng',0.0)),
            'vicinity': rec.get('vicinity',rec.get('row',{}).get('vicinity','')),
            'types': rec.get('types',rec.get('row',{}).get('types',[])),
            'source': source
        }

    def _fetch_poorly_rated(self, token):
        bad = set()
        if not token:
            return bad
        try:
            BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8080')
            resp = requests.get(f"{BACKEND_URL}/api/reviews/",
                                headers={"Authorization":f"Bearer {token}"})
            if resp.status_code==200:
                for rev in resp.json():
                    if float(rev.get('rating',0))<=2.0:
                        bad.add(str(rev.get('place_id')))
        except Exception:
            pass
        return bad

    def _prepare_metalearner_features(self, recs_by_source):
        """Prepare features for the metalearner from individual model predictions."""
        features = []
        for source in self.model_features:
            source_recs = recs_by_source.get(source, [])
            if source_recs:
                features.append(np.mean([r['score'] for r in source_recs]))
            else:
                features.append(0.0)
        return np.array(features).reshape(1, -1)

    def train_metalearner(self, user_feedback):
        """
        Train the metalearner using user feedback.
        user_feedback: List of dicts with keys:
            - place_id: str
            - rating: float (1-5)
            - source_predictions: dict mapping source names to their original scores
        """
        if not user_feedback:
            return

        X = []
        y = []
        
        for feedback in user_feedback:
            features = []
            for source in self.model_features:
                features.append(feedback['source_predictions'].get(source, 0.0))
            X.append(features)
            y.append(feedback['rating'])

        X = np.array(X)
        y = np.array(y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train metalearner
        self.metalearner.fit(X_scaled, y)
        self.is_metalearner_trained = True

    def _get_metalearner_weights(self, recs_by_source):
        """Get weights from metalearner for current predictions."""
        if not self.is_metalearner_trained:
            return self.weights

        features = self._prepare_metalearner_features(recs_by_source)
        
        # Get coefficients and normalize to sum to 1
        weights = self.metalearner.coef_
        weights = np.maximum(weights, 0)  # Ensure non-negative weights
        weights = weights / weights.sum() if weights.sum() > 0 else self.weights
        
        return dict(zip(self.model_features, weights))

    def get_recommendations(self, user_lat, user_lon, user_prefs, predicted_ratings_dict,
                            num_recs=10, user_token=None):
        # Determine poorly rated to exclude
        poorly_rated = self._fetch_poorly_rated(user_token)
        target = num_recs * 3
        all_recs = []
        recs_by_source = defaultdict(list)
        
        # Helper to collect from each model
        def collect(model, recs, source):
            if not recs:
                return
            normalized = self._normalize_scores(recs)
            recs_by_source[source].extend(normalized)
            all_recs.extend([self._standardize(r, source) for r in normalized])

        # Get recommendations from each model
        # Autoencoder
        if self.autoencoder_recommender:
            cats = list(predicted_ratings_dict.keys())
            arr = np.array([predicted_ratings_dict.get(c,0) for c in cats])
            mask = np.array([c in user_prefs and user_prefs[c]>0 for c in cats])
            recs = self.autoencoder_recommender.get_recommendations(
                user_lat, user_lon, arr, mask, target)
            collect(self.autoencoder_recommender, recs, 'autoencoder')

        # SVD
        if self.svd_recommender:
            df = self.autoencoder_recommender.places_df if self.autoencoder_recommender else None
            if df is not None:
                recs = self.svd_recommender.get_recommendations(
                    df, user_lat, user_lon, predicted_ratings_dict, target)
                collect(self.svd_recommender, recs, 'svd')

        # Transfer
        if self.transfer_recommender and df is not None:
            recs = self.transfer_recommender.get_recommendations(
                user_prefs, user_lat, user_lon, df, target)
            collect(self.transfer_recommender, recs, 'transfer')

        # Madrid transfer
        if self.madrid_transfer_recommender:
            recs = self.madrid_transfer_recommender.get_recommendations(
                user_lat, user_lon, user_prefs, target)
            collect(self.madrid_transfer_recommender, recs, 'madrid_transfer')

        # Get dynamic weights from metalearner
        current_weights = self._get_metalearner_weights(recs_by_source)
        
        # Apply weights
        for r in all_recs:
            r['score'] *= current_weights.get(r['source'], 1.0)

        # Apply boosts & penalties
        all_recs = self._apply_distance_penalty(all_recs)
        all_recs = self._apply_diversity_boost(all_recs)
        all_recs = self._apply_novelty_boost(all_recs)

        # Deduplicate by highest score
        unique = {}
        for r in sorted(all_recs, key=lambda x: x['score'], reverse=True):
            pid = r['place_id']
            if pid and pid not in unique:
                unique[pid] = r

        # Filter out poorly rated and select top N
        final = []
        for r in unique.values():
            pid = str(r['place_id'])
            if pid in poorly_rated:
                continue
            final.append(r)
            if len(final)==num_recs:
                break

        return final
