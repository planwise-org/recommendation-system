import unittest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the Python path to be able to import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Add streamlit directory to path to allow imports from there
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath('planwise')))

# Now import the module
from planwise.src.recommenders.transfer_recommender import TransferRecommender

class TestTransferRecommender(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a test instance with reduced embedding dimension for faster tests
        self.recommender = TransferRecommender(embedding_dim=16)
        
        # Create a simple mock dataframe for testing
        self.test_df = pd.DataFrame({
            'place_id': ['1', '2', '3', '4', '5'],
            'name': ['Place 1', 'Place 2', 'Place 3', 'Place 4', 'Place 5'],
            'lat': [40.416775, 40.417891, 40.415764, 40.420123, 40.412123],
            'lng': [-3.703790, -3.701234, -3.704532, -3.705678, -3.702789],
            'rating': [4.5, 3.8, 4.2, 3.5, 4.7],
            'user_ratings_total': [100, 50, 75, 30, 120],
            'types': ['restaurant, bar', 'museum', 'park, tourist_attraction', 'cafe', 'lodging'],
            'description': ['Description 1', 'Description 2', 'Description 3', 'Description 4', 'Description 5'],
            'vicinity': ['Address 1', 'Address 2', 'Address 3', 'Address 4', 'Address 5'],
            'icon': ['icon1.png', 'icon2.png', 'icon3.png', 'icon4.png', 'icon5.png']
        })
        
        # Mock user preferences
        self.user_preferences = {
            "restaurants": 4.0,
            "museums": 4.5,
            "parks": 3.5,
            "cafes": 2.5,
            "hotels/other lodgings": 3.0,
            "pubs/bars": 3.8
        }
        
        # User location for tests
        self.user_lat = 40.416775
        self.user_lng = -3.703790
        
        # Create fake embeddings for testing
        self.recommender.movie_embeddings = np.random.normal(0, 0.1, size=(100, self.recommender.embedding_dim))
        self.recommender.movie_ids = np.arange(100)
        self.recommender.place_embeddings = {
            '1': np.random.normal(0, 0.1, size=self.recommender.embedding_dim),
            '2': np.random.normal(0, 0.1, size=self.recommender.embedding_dim),
            '3': np.random.normal(0, 0.1, size=self.recommender.embedding_dim),
            '4': np.random.normal(0, 0.1, size=self.recommender.embedding_dim),
            '5': np.random.normal(0, 0.1, size=self.recommender.embedding_dim)
        }
    
    def test_haversine_distance(self):
        """Test the haversine distance calculation function."""
        lat1, lon1 = 40.416775, -3.703790  # Madrid center
        lat2, lon2 = 40.417891, -3.701234  # Near location
        
        expected_distance = 0.241  # km (approximate)
        actual_distance = self.recommender._haversine_distance(lat1, lon1, lat2, lon2)
        
        # Test that the calculated distance is close to the expected value
        self.assertAlmostEqual(actual_distance, expected_distance, delta=0.01)
        
        # Test that distance from point to itself is 0
        self.assertEqual(self.recommender._haversine_distance(lat1, lon1, lat1, lon1), 0)
    
    def test_preferences_to_embedding(self):
        """Test conversion of user preferences to embeddings."""
        # Test with valid preferences
        embedding = self.recommender._preferences_to_embedding(self.user_preferences)
        
        # Check that embedding has the correct shape
        self.assertEqual(embedding.shape, (self.recommender.embedding_dim,))
        
        # Test with empty preferences
        embedding = self.recommender._preferences_to_embedding({})
        self.assertEqual(embedding.shape, (self.recommender.embedding_dim,))
    
    @patch('planwise.src.recommenders.transfer_recommender.TransferRecommender._preferences_to_embedding')
    def test_get_recommendations(self, mock_pref_to_embedding):
        """Test recommendation generation."""
        # Mock the preferences to embedding function
        mock_embedding = np.random.normal(0, 0.1, size=self.recommender.embedding_dim)
        mock_pref_to_embedding.return_value = mock_embedding
        
        # Get recommendations
        recommendations = self.recommender.get_recommendations(
            self.user_preferences,
            self.user_lat,
            self.user_lng,
            self.test_df,
            top_n=3
        )
        
        # Verify that the preferences to embedding function was called
        mock_pref_to_embedding.assert_called_once_with(self.user_preferences)
        
        # Check that we got recommendations (might be empty if no matches)
        # Typically this would return recommendations, but our mock data might not match
        # the criteria in the actual implementation
        self.assertIsInstance(recommendations, list)
        
        # If we got recommendations, check their structure
        if recommendations:
            expected_fields = ['place_id', 'name', 'score', 'distance', 'lat', 'lng']
            for field in expected_fields:
                self.assertTrue(all(field in rec for rec in recommendations))
            
            # Check that recommendations are sorted by score in descending order
            scores = [rec['score'] for rec in recommendations]
            self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_get_movies_by_genre(self):
        """Test the function that gets movies by genre."""
        indices = self.recommender._get_movies_by_genre(None)
        
        # Check that we get indices
        self.assertTrue(len(indices) > 0)
        
        # Check that indices are within range
        self.assertTrue(all(0 <= idx < 100 for idx in indices))
    
    @patch('os.path.exists')
    @patch('joblib.dump')
    def test_transfer_to_places(self, mock_dump, mock_exists):
        """Test the transfer learning process."""
        # Mock the exists function to return False so we execute the transfer code
        mock_exists.return_value = False
        
        # Call the transfer function
        self.recommender.transfer_to_places(self.test_df, save_path="mock_path")
        
        # Check that the place embeddings were created
        self.assertTrue(hasattr(self.recommender, 'place_embeddings'))
        
        # Check that embeddings were created for each place
        for place_id in self.test_df['place_id']:
            self.assertIn(place_id, self.recommender.place_embeddings)
        
        # Check that joblib.dump was called to save the embeddings
        mock_dump.assert_called_once()

if __name__ == '__main__':
    unittest.main() 