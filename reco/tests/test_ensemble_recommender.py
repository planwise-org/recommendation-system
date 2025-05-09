import unittest
import pandas as pd
import math
from unittest.mock import patch, MagicMock, Mock

# Now import the module
from planwise.src.recommenders import EnsembleRecommender

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

        # Define the haversine function for testing
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371000  # Earth's radius in meters
            phi1 = math.radians(lat1)
            phi2 = math.radians(lat2)
            dphi = math.radians(lat2 - lat1)
            dlambda = math.radians(lon2 - lon1)
            a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            return R * c
        
        # Attach haversine function to recommender for testing
        self.haversine = haversine
    
    def test_normalize_scores(self):
        """Test the score normalization function."""
        # Test with regular recommendations
        normalized = self.recommender._normalize_scores(self.auto_recs)
        
        # Check that scores are normalized between 0 and 1
        for rec in normalized:
            self.assertTrue(0 <= rec['score'] <= 1)
            self.assertTrue('original_score' in rec)
        
        # Check that the highest score is normalized to 1 or close to it
        highest_score_rec = max(normalized, key=lambda x: x['score'])
        self.assertAlmostEqual(highest_score_rec['score'], 1.0, places=1)
        
        # Test with empty list
        normalized = self.recommender._normalize_scores([])
        self.assertEqual(normalized, [])
        
        # Test with single item
        single_rec = [{'place_id': '1', 'name': 'Place 1', 'score': 0.85}]
        normalized = self.recommender._normalize_scores(single_rec)
        # A single item should get a score in the middle of the range
        if len(normalized) > 0:
            self.assertTrue(0 <= normalized[0]['score'] <= 1)
    
    def test_distance_calculation(self):
        """Test distance calculations."""
        lat1, lon1 = 40.416775, -3.703790  # Madrid center
        lat2, lon2 = 40.417891, -3.701234  # Near location
        
        # Apply distance penalty to recommendations
        recs = self.auto_recs.copy()
        penalized = self.recommender._apply_distance_penalty(recs)
        
        # Check that distance factors are applied
        for rec in penalized:
            self.assertTrue('distance_factor' in rec)
            self.assertTrue(0 < rec['distance_factor'] <= 1.0)
    
    @patch('planwise.src.recommenders.ensemble_recommender.EnsembleRecommender.initialize_models')
    @patch('planwise.src.recommenders.AutoencoderRecommender.get_recommendations')
    @patch('planwise.src.recommenders.SVDPlaceRecommender.get_recommendations')
    @patch('planwise.src.recommenders.TransferRecommender.get_recommendations')
    @patch('planwise.src.recommenders.MadridTransferRecommender.get_recommendations')
    def test_get_recommendations(self, mock_madrid, mock_transfer, mock_svd, mock_auto, mock_init):
        """Test the main recommendation function."""
        # Configure mocks
        mock_auto.return_value = self.auto_recs
        mock_svd.return_value = self.svd_recs
        mock_transfer.return_value = self.transfer_recs
        mock_madrid.return_value = []
        
        # Setup recommender
        self.recommender.autoencoder_recommender = MagicMock()
        self.recommender.svd_recommender = MagicMock()
        self.recommender.transfer_recommender = MagicMock()
        self.recommender.madrid_transfer_recommender = MagicMock()
        self.recommender.autoencoder_recommender.get_recommendations.return_value = self.auto_recs
        self.recommender.svd_recommender.get_recommendations.return_value = self.svd_recs
        self.recommender.transfer_recommender.get_recommendations.return_value = self.transfer_recs
        self.recommender.madrid_transfer_recommender.get_recommendations.return_value = []
        self.recommender.autoencoder_recommender.places_df = self.test_df
        
        # Get recommendations
        recommendations = self.recommender.get_recommendations(
            self.user_lat, 
            self.user_lng,
            self.user_prefs,
            self.predicted_ratings_dict,
            num_recs=3
        )
        
        # Check that we got recommendations
        self.assertTrue(isinstance(recommendations, list))
        
        # If we have recommendations, check their structure
        if recommendations:
            expected_fields = ['place_id', 'name', 'score', 'source']
            for field in expected_fields:
                self.assertTrue(all(field in rec for rec in recommendations))
        
    @patch('planwise.src.recommenders.ensemble_recommender.EnsembleRecommender.initialize_models')
    def test_empty_recommendations(self, mock_init):
        """Test behavior with empty recommendation lists."""
        # Setup recommender with empty results
        self.recommender.autoencoder_recommender = MagicMock()
        self.recommender.svd_recommender = MagicMock()
        self.recommender.transfer_recommender = MagicMock()
        self.recommender.madrid_transfer_recommender = MagicMock()
        self.recommender.autoencoder_recommender.get_recommendations.return_value = []
        self.recommender.svd_recommender.get_recommendations.return_value = []
        self.recommender.transfer_recommender.get_recommendations.return_value = []
        self.recommender.madrid_transfer_recommender.get_recommendations.return_value = []
        self.recommender.autoencoder_recommender.places_df = self.test_df
        
        # Get recommendations
        recommendations = self.recommender.get_recommendations(
            self.user_lat,
            self.user_lng,
            self.user_prefs,
            self.predicted_ratings_dict,
            num_recs=3
        )
        
        # Check that we got no recommendations
        self.assertEqual(len(recommendations), 0)

if __name__ == '__main__':
    unittest.main() 