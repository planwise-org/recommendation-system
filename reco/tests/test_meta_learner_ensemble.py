import unittest
import pandas as pd
from unittest.mock import patch, MagicMock, Mock

# Now import the module
from planwise.src.recommenders import MetaEnsembleRecommender

class TestMetaEnsembleRecommender(unittest.TestCase):
    """Test suite for the MetaEnsembleRecommender class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create custom weights for testing
        self.weights = {
            'autoencoder': 0.25,
            'svd': 0.25,
            'transfer': 0.25,
            'madrid_transfer': 0.25
        }
        
        # Create the recommender instance with custom weights
        self.recommender = MetaEnsembleRecommender(weights=self.weights)
        
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

        self.madrid_transfer_recs = [
            {'place_id': '5', 'name': 'Place 5', 'score': 0.95, 'lat': 40.412123, 'lng': -3.702789, 'distance': 2000},
            {'place_id': '2', 'name': 'Place 2', 'score': 0.85, 'lat': 40.417891, 'lng': -3.701234, 'distance': 1500}
        ]

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
        if len(normalized) > 0:
            self.assertTrue(0 <= normalized[0]['score'] <= 1)

    def test_distance_penalty(self):
        """Test the distance penalty application."""
        recs = self.auto_recs.copy()
        # Store original scores before applying penalty
        for rec in recs:
            rec['original_score'] = rec['score']
            
        penalized = self.recommender._apply_distance_penalty(recs)
        
        # Check that distance factors are applied
        for rec in penalized:
            self.assertTrue('distance_factor' in rec)
            self.assertTrue(0 < rec['distance_factor'] <= 1.0)
            # Check that scores are reduced by distance
            self.assertLessEqual(rec['score'], rec['original_score'])

    def test_diversity_boost(self):
        """Test the diversity boost application."""
        recs = self.auto_recs.copy()
        for r in recs:
            r['category'] = 'restaurant'  # Set same category for testing
        
        boosted = self.recommender._apply_diversity_boost(recs)
        
        # Check that scores are adjusted for diversity
        self.assertGreater(boosted[0]['score'], boosted[1]['score'])

    def test_novelty_boost(self):
        """Test the novelty boost application."""
        recs = [
            {'score': 0.8, 'user_ratings_total': 5},  # Low reviews
            {'score': 0.8, 'user_ratings_total': 100}  # High reviews
        ]
        
        boosted = self.recommender._apply_novelty_boost(recs)
        
        # Check that low-review items get a boost
        self.assertGreater(boosted[0]['score'], boosted[1]['score'])
        self.assertTrue('novelty_boost' in boosted[0])

    def test_metalearner_training(self):
        """Test the metalearner training functionality."""
        # Create mock user feedback
        user_feedback = [
            {
                'place_id': '1',
                'rating': 4.5,
                'source_predictions': {
                    'autoencoder': 0.8,
                    'svd': 0.7,
                    'transfer': 0.75,
                    'madrid_transfer': 0.85
                }
            },
            {
                'place_id': '2',
                'rating': 3.0,
                'source_predictions': {
                    'autoencoder': 0.6,
                    'svd': 0.5,
                    'transfer': 0.55,
                    'madrid_transfer': 0.65
                }
            }
        ]
        
        # Train the metalearner
        self.recommender.train_metalearner(user_feedback)
        
        # Check that metalearner is trained
        self.assertTrue(self.recommender.is_metalearner_trained)
        self.assertIsNotNone(self.recommender.metalearner.coef_)

    @patch('planwise.src.recommenders.meta_learner_ensemble.MetaEnsembleRecommender.initialize_models')
    def test_get_recommendations(self, mock_init):
        """Test the main recommendation function."""
        # Setup recommender with mock models
        self.recommender.autoencoder_recommender = MagicMock()
        self.recommender.svd_recommender = MagicMock()
        self.recommender.transfer_recommender = MagicMock()
        self.recommender.madrid_transfer_recommender = MagicMock()
        
        # Configure mock return values
        self.recommender.autoencoder_recommender.get_recommendations.return_value = self.auto_recs
        self.recommender.svd_recommender.get_recommendations.return_value = self.svd_recs
        self.recommender.transfer_recommender.get_recommendations.return_value = self.transfer_recs
        self.recommender.madrid_transfer_recommender.get_recommendations.return_value = self.madrid_transfer_recs
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
            
            # Check that recommendations are unique
            place_ids = [rec['place_id'] for rec in recommendations]
            self.assertEqual(len(place_ids), len(set(place_ids)))

    def test_standardize_recommendation(self):
        """Test the recommendation standardization function."""
        test_rec = {
            'place_id': '1',
            'name': 'Test Place',
            'score': 0.8,
            'category': 'restaurant',
            'rating': 4.5,
            'user_ratings_total': 100,
            'distance': 500,
            'lat': 40.416775,
            'lng': -3.703790,
            'types': ['restaurant', 'bar']
        }
        
        standardized = self.recommender._standardize(test_rec, 'test_source')
        
        # Check all required fields are present
        required_fields = [
            'place_id', 'name', 'score', 'category', 'icon',
            'rating', 'user_ratings_total', 'distance',
            'lat', 'lng', 'vicinity', 'types', 'source'
        ]
        for field in required_fields:
            self.assertIn(field, standardized)
        
        # Check source is set correctly
        self.assertEqual(standardized['source'], 'test_source')

if __name__ == '__main__':
    unittest.main() 