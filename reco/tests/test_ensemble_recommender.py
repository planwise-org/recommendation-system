import unittest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

# Add the parent directory to the Python path to be able to import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Add streamlit directory to path to allow imports from there
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath('streamlit')))

# Mock streamlit before importing it
import streamlit as st
st.session_state = {}

# Now import the module
from streamlit.src.recommenders.ensemble_recommender import EnsembleRecommender

class TestEnsembleRecommender(unittest.TestCase):
    """Test suite for the EnsembleRecommender class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create custom weights for testing
        self.weights = {
            'autoencoder': 0.4,
            'svd': 0.3,
            'transfer': 0.3
        }
        
        # Create the recommender instance with custom weights
        self.recommender = EnsembleRecommender(weights=self.weights)
        
        # Create a simple mock dataframe for testing
        self.test_df = pd.DataFrame({
            'place_id': ['1', '2', '3', '4', '5'],
            'name': ['Place 1', 'Place 2', 'Place 3', 'Place 4', 'Place 5'],
            'lat': [40.416775, 40.417891, 40.415764, 40.420123, 40.412123],
            'lng': [-3.703790, -3.701234, -3.704532, -3.705678, -3.702789],
            'rating': [4.5, 3.8, 4.2, 3.5, 4.7],
            'user_ratings_total': [100, 50, 75, 30, 120],
            'types': ['restaurant, bar', 'museum', 'park, tourist_attraction', 'cafe', 'lodging'],
            'types_processed': [['restaurant', 'bar'], ['museum'], ['park', 'tourist_attraction'], 
                                ['cafe'], ['lodging']]
        })
        
        # Mock user preferences and predicted ratings
        self.user_prefs = {
            "restaurants": 4.0,
            "museums": 4.5,
            "parks": 3.5,
            "cafes": 2.5,
            "hotels/other lodgings": 3.0,
            "pubs/bars": 3.8
        }
        
        self.predicted_ratings_dict = self.user_prefs.copy()
        
        # User location for tests
        self.user_lat = 40.416775
        self.user_lng = -3.703790
        
        # Create mock recommendations from each model
        self.auto_recs = [
            {'place_id': '1', 'name': 'Place 1', 'score': 0.85, 'lat': 40.416775, 'lng': -3.703790, 'distance': 100},
            {'place_id': '3', 'name': 'Place 3', 'score': 0.75, 'lat': 40.415764, 'lng': -3.704532, 'distance': 800}
        ]
        
        self.svd_recs = [
            {'place_id': '2', 'name': 'Place 2', 'score': 0.90, 'lat': 40.417891, 'lng': -3.701234, 'distance': 1500},
            {'place_id': '1', 'name': 'Place 1', 'score': 0.70, 'lat': 40.416775, 'lng': -3.703790, 'distance': 100}
        ]
        
        self.transfer_recs = [
            {'place_id': '4', 'name': 'Place 4', 'score': 0.80, 'lat': 40.420123, 'lng': -3.705678, 'distance': 1200},
            {'place_id': '3', 'name': 'Place 3', 'score': 0.65, 'lat': 40.415764, 'lng': -3.704532, 'distance': 800}
        ]

        # Patch the transfer recommender import
        self.transfer_recommender_patch = patch('streamlit.src.ensemble.TransferRecommender')
        self.mock_transfer_recommender = self.transfer_recommender_patch.start()
    
    def tearDown(self):
        """Clean up after each test."""
        self.transfer_recommender_patch.stop()
    
    def test_normalize_scores(self):
        """Test the score normalization function."""
        # Test with regular recommendations
        normalized = self.recommender._normalize_scores(self.auto_recs)
        
        # Check that scores are normalized between 0 and 1
        for rec in normalized:
            self.assertTrue(0 <= rec['normalized_score'] <= 1)
        
        # Check that the highest score is normalized to 1
        highest_score_rec = max(normalized, key=lambda x: x['score'])
        self.assertEqual(highest_score_rec['normalized_score'], 1.0)
        
        # Test with empty list
        normalized = self.recommender._normalize_scores([])
        self.assertEqual(normalized, [])
        
        # Test with single item
        single_rec = [{'place_id': '1', 'name': 'Place 1', 'score': 0.85}]
        normalized = self.recommender._normalize_scores(single_rec)
        self.assertEqual(normalized[0]['normalized_score'], 1.0)
    
    def test_haversine(self):
        """Test the haversine distance calculation function."""
        lat1, lon1 = 40.416775, -3.703790  # Madrid center
        lat2, lon2 = 40.417891, -3.701234  # Near location
        
        expected_distance = 241.0  # meters (approximate)
        actual_distance = self.recommender.haversine(lat1, lon1, lat2, lon2)
        
        # Test that the calculated distance is close to the expected value
        self.assertAlmostEqual(actual_distance, expected_distance, delta=10.0)
        
        # Test that distance from point to itself is 0
        self.assertEqual(self.recommender.haversine(lat1, lon1, lat1, lon1), 0)
    
    @patch('streamlit.src.recommenders.ensemble_recommender.EnsembleRecommender._get_autoencoder_recommendations')
    @patch('streamlit.src.recommenders.ensemble_recommender.EnsembleRecommender._get_svd_recommendations')
    @patch('streamlit.src.recommenders.ensemble_recommender.EnsembleRecommender._get_transfer_recommendations')
    def test_get_recommendations(self, mock_transfer, mock_svd, mock_auto):
        """Test the main recommendation function."""
        # Configure the mocks to return our test recommendations
        mock_auto.return_value = self.auto_recs
        mock_svd.return_value = self.svd_recs
        mock_transfer.return_value = self.transfer_recs
        
        # Get recommendations
        recommendations = self.recommender.get_recommendations(
            self.user_lat,
            self.user_lng,
            self.user_prefs,
            self.predicted_ratings_dict,
            num_recs=3
        )
        
        # Verify that all model recommendation functions were called
        mock_auto.assert_called_once_with(self.user_lat, self.user_lng, self.user_prefs, 6)
        mock_svd.assert_called_once_with(self.user_lat, self.user_lng, self.predicted_ratings_dict, 6)
        mock_transfer.assert_called_once_with(self.user_lat, self.user_lng, self.predicted_ratings_dict, 6)
        
        # Check that we got recommendations
        self.assertEqual(len(recommendations), 3)
        
        # Check that recommendations have the expected fields
        expected_fields = ['place_id', 'name', 'score', 'ensemble_score', 'sources']
        for field in expected_fields:
            self.assertTrue(all(field in rec for rec in recommendations))
        
        # Check that recommendations are sorted by ensemble score in descending order
        scores = [rec['ensemble_score'] for rec in recommendations]
        self.assertEqual(scores, sorted(scores, reverse=True))
        
        # Check that Places 1 (which appears in both auto and svd) 
        # has both sources listed in its 'sources' field
        place1 = next((r for r in recommendations if r['place_id'] == '1'), None)
        if place1:
            self.assertTrue(set(place1['sources']).issuperset({'autoencoder', 'svd'}))
    
    @patch('streamlit.src.recommenders.ensemble_recommender.EnsembleRecommender._get_autoencoder_recommendations')
    @patch('streamlit.src.recommenders.ensemble_recommender.EnsembleRecommender._get_svd_recommendations')
    @patch('streamlit.src.recommenders.ensemble_recommender.EnsembleRecommender._get_transfer_recommendations')
    def test_empty_recommendations(self, mock_transfer, mock_svd, mock_auto):
        """Test behavior with empty recommendation lists."""
        # Configure the mocks to return empty lists
        mock_auto.return_value = []
        mock_svd.return_value = []
        mock_transfer.return_value = []
        
        # Get recommendations
        recommendations = self.recommender.get_recommendations(
            self.user_lat,
            self.user_lng,
            self.user_prefs,
            self.predicted_ratings_dict,
            num_recs=3
        )
        
        # Check that we got an empty list
        self.assertEqual(recommendations, [])

if __name__ == '__main__':
    unittest.main() 