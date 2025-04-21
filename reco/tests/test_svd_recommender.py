import unittest
import pandas as pd
import numpy as np
import sys
import os
import math
from pathlib import Path

# Add the parent directory to the Python path to be able to import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Add streamlit directory to path to allow imports from there
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath('planwise')))

# Now import the module
from planwise.src.recommenders.svd_recommender import SVDPlaceRecommender

class TestSVDPlaceRecommender(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a basic category to place types mapping
        self.category_to_place_types = {
            "restaurants": ["restaurant"],
            "museums": ["museum"],
            "parks": ["park"],
            "cafes": ["cafe"],
            "hotels/other lodgings": ["lodging"],
            "pubs/bars": ["bar"]
        }
        
        # Initialize the recommender with the mapping
        self.recommender = SVDPlaceRecommender(category_to_place_types=self.category_to_place_types)
        
        # Make sure category_to_place_types is set in the recommender
        if not self.recommender.category_to_place_types:
            self.recommender.category_to_place_types = self.category_to_place_types
        
        # Create a simple mock dataframe for testing
        self.test_df = pd.DataFrame({
            'place_id': ['1', '2', '3', '4', '5'],
            'name': ['Place 1', 'Place 2', 'Place 3', 'Place 4', 'Place 5'],
            'lat': [40.416775, 40.417891, 40.415764, 40.420123, 40.412123],
            'lng': [-3.703790, -3.701234, -3.704532, -3.705678, -3.702789],
            'rating': [4.5, 3.8, 4.2, 3.5, 4.7],
            'user_ratings_total': [100, 50, 75, 30, 120],
            'types': ['restaurant, bar', 'museum', 'park, tourist_attraction', 'cafe', 'lodging'],
            'vicinity': ['Address 1', 'Address 2', 'Address 3', 'Address 4', 'Address 5']
        })
        
        # Mock predicted ratings
        self.predicted_ratings = {
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
    
    def test_haversine_distance(self):
        """Test the haversine distance calculation function."""
        # Test with known coordinates
        lat1, lon1 = 40.416775, -3.703790  # Madrid center (Puerta del Sol)
        lat2, lon2 = 40.417891, -3.701234  # Near location
        
        expected_distance = 0.241  # km (approximate)
        actual_distance = self.recommender.haversine_distance(lat1, lon1, lat2, lon2)
        
        # Test that the calculated distance is close to the expected value
        self.assertAlmostEqual(actual_distance, expected_distance, delta=0.01)
        
        # Test that distance from point to itself is 0
        self.assertEqual(self.recommender.haversine_distance(lat1, lon1, lat1, lon1), 0)
    
    def test_get_type_score(self):
        """Test the type scoring functionality."""
        # Test with a place type that matches user preferences
        place_types = "restaurant, bar"
        score = self.recommender.get_type_score(place_types, self.predicted_ratings)
        self.assertGreater(score, 0)
        
        # Test with a place type that doesn't match user preferences
        place_types = "unknown_type"
        score = self.recommender.get_type_score(place_types, self.predicted_ratings)
        self.assertEqual(score, 0)
        
        # Test with None/NaN
        score = self.recommender.get_type_score(pd.NA, self.predicted_ratings)
        self.assertEqual(score, 0)
    
    def test_prepare_data(self):
        """Test the data preparation for SVD."""
        data = self.recommender.prepare_data(self.test_df)
        
        # Check that the data has been properly prepared
        self.assertIsNotNone(data)
        
        # Check that trainset can be built
        trainset = data.build_full_trainset()
        self.assertEqual(trainset.n_items, len(self.test_df))
    
    def test_fit_and_predict(self):
        """Test fitting the model and making predictions."""
        # Fit the model
        self.recommender.fit(self.test_df)
        
        # Get recommendations
        recommendations = self.recommender.get_recommendations(
            self.test_df, 
            self.user_lat, 
            self.user_lng, 
            self.predicted_ratings, 
            top_n=3,
            max_distance=10
        )
        
        # Check that we got recommendations
        self.assertEqual(len(recommendations), 3)
        
        # Print the actual fields in the first recommendation to debug
        if recommendations:
            print("\nActual fields in recommendation:", list(recommendations[0].keys()))
        
        # Get the actual fields from the first recommendation and check each recommendation
        # contains these fields
        if recommendations:
            actual_fields = set(recommendations[0].keys())
            for rec in recommendations:
                self.assertEqual(set(rec.keys()), actual_fields)
            
            # Check for minimum required fields - these should definitely exist
            min_required_fields = ['place_id', 'name', 'score', 'distance']
            for field in min_required_fields:
                self.assertTrue(all(field in rec for rec in recommendations))
        
        # Check that recommendations are sorted by score in descending order
        scores = [rec['score'] for rec in recommendations]
        self.assertEqual(scores, sorted(scores, reverse=True))

if __name__ == '__main__':
    unittest.main() 