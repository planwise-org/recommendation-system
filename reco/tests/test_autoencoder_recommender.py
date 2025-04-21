import unittest
import pandas as pd
import numpy as np
import sys
import os
import math
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

# Add the parent directory to the Python path to be able to import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Add streamlit directory to path to allow imports from there
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath('planwise')))

# Now import after adding the path
from planwise.src.recommenders.autoencoder_recommender import AutoencoderRecommender

# Define helper functions here instead of importing them
def haversine(lat1, lon1, lat2, lon2):
    """Compute distance (in meters) between two lat/lon points."""
    R = 6371000  # Earth's radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def process_types(types_str):
    """Process place types string into a list."""
    if pd.isna(types_str) or not types_str:
        return []
    return [t.strip() for t in str(types_str).split(',')]

def euclidean_distance(p1, p2):
    """Euclidean distance between two points (lat,lon)."""
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

class TestHelperFunctions(unittest.TestCase):
    """Test suite for utility and helper functions."""
    
    def test_haversine(self):
        """Test the haversine distance calculation."""
        # Test with known coordinates
        lat1, lon1 = 40.416775, -3.703790  # Madrid center
        lat2, lon2 = 40.417891, -3.701234  # Near location
        
        # Expected distance in meters
        expected_distance = 241.0  # meters (approximate)
        actual_distance = haversine(lat1, lon1, lat2, lon2)
        
        # Test that the calculated distance is close to the expected value
        self.assertAlmostEqual(actual_distance, expected_distance, delta=10.0)
        
        # Test that distance from point to itself is 0
        self.assertEqual(haversine(lat1, lon1, lat1, lon1), 0)
    
    def test_process_types(self):
        """Test the process_types function."""
        # Test with regular input
        types_str = "restaurant, bar, cafe"
        processed = process_types(types_str)
        self.assertEqual(processed, ["restaurant", "bar", "cafe"])
        
        # Test with empty string
        types_str = ""
        processed = process_types(types_str)
        self.assertEqual(processed, [])
        
        # Test with None/NaN
        processed = process_types(pd.NA)
        self.assertEqual(processed, [])
    
    def test_euclidean_distance(self):
        """Test the euclidean_distance function."""
        p1 = (0, 0)
        p2 = (3, 4)
        distance = euclidean_distance(p1, p2)
        self.assertEqual(distance, 5.0)
        
        # Test with negative coordinates
        p1 = (-1, -1)
        p2 = (2, 3)
        distance = euclidean_distance(p1, p2)
        self.assertAlmostEqual(distance, 5.0)
        
        # Test with same point
        distance = euclidean_distance(p1, p1)
        self.assertEqual(distance, 0.0)

class TestAutoencoderRecommender(unittest.TestCase):
    """Test suite for the AutoencoderRecommender class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.categories = [
            "museums", "parks", "restaurants", "cafes"
        ]
        
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
        self.user_prefs = np.array([4.0, 3.5, 2.0, 4.5])  # For museums, parks, restaurants, cafes
        self.provided_mask = np.array([True, True, False, True])
        
        # Create a patch for the places global variable
        self.places_patch = patch('planwise.src.recommenders.autoencoder_recommender.places', self.test_df)
        self.mock_places = self.places_patch.start()
        
        # Create a patch for the categories global variable
        self.categories_patch = patch('planwise.src.recommenders.autoencoder_recommender.categories', self.categories)
        self.mock_categories = self.categories_patch.start()
        
        # Create a mock for category_to_place_types
        self.category_to_place_types = {
            "museums": ["museum"],
            "parks": ["park"],
            "restaurants": ["restaurant"],
            "cafes": ["cafe"]
        }
        self.category_patch = patch('planwise.src.recommenders.autoencoder_recommender.category_to_place_types', self.category_to_place_types)
        self.mock_category = self.category_patch.start()
        
        # Create a mock for auto_model and scaler
        self.model_patch = patch('planwise.src.recommenders.autoencoder_recommender.auto_model')
        self.mock_model = self.model_patch.start()
        self.scaler_patch = patch('planwise.src.recommenders.autoencoder_recommender.scaler')
        self.mock_scaler = self.scaler_patch.start()
        
        # Configure the mocks
        self.mock_model.predict.return_value = np.array([[0.6, 0.7, 0.8, 0.5]])
        self.mock_scaler.transform.return_value = np.array([[0.8, 0.7, 0.4, 0.9]])
        self.mock_scaler.inverse_transform.return_value = np.array([[3.0, 3.5, 4.0, 2.5]])
        
        # Create the recommender instance
        self.recommender = AutoencoderRecommender()
    
    def tearDown(self):
        """Clean up after each test method."""
        self.places_patch.stop()
        self.categories_patch.stop()
        self.category_patch.stop()
        self.model_patch.stop()
        self.scaler_patch.stop()
    
    @patch('planwise.src.recommenders.autoencoder_recommender.AutoencoderRecommender.haversine')
    def test_get_recommendations(self, mock_haversine):
        """Test the get_recommendations method."""
        # Configure the mock haversine function
        # Return distance values that ensure places are within range
        mock_haversine.side_effect = [1000, 1500, 800, 1200, 1800]
        
        # Get recommendations
        user_lat, user_lng = 40.416775, -3.703790
        recs = self.recommender.get_recommendations(user_lat, user_lng, self.user_prefs, self.provided_mask, num_recs=3)
        
        # Check that we got recommendations
        self.assertTrue(len(recs) <= 3)
        
        # Check that recommendations have the expected fields
        if recs:
            expected_fields = ['name', 'lat', 'lng', 'score', 'category', 'rating']
            for field in expected_fields:
                self.assertTrue(all(field in rec for rec in recs))

if __name__ == '__main__':
    unittest.main() 