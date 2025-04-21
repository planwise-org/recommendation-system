# Unit Testing

## Overview

Unit testing is the foundation of our testing strategy, focusing on testing individual components in isolation. This approach helps us identify issues early, maintain code quality, and make refactoring safer.

## Unit Test Philosophy

Our unit testing philosophy follows these principles:

1. **Isolation**: Each unit test should test a single unit of functionality in isolation
2. **Independence**: Tests should not depend on each other or external state
3. **Completeness**: Cover normal cases, edge cases, and error cases
4. **Clarity**: Tests should be clear about what they're testing
5. **Speed**: Unit tests should run quickly to encourage frequent testing

## Test Structure

We use the standard unittest framework with a class-based approach:

```python
import unittest
from unittest.mock import patch, MagicMock

from src.recommenders.svd_recommender import SVDPlaceRecommender

class TestSVDPlaceRecommender(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.recommender = SVDPlaceRecommender(category_to_place_types={})
        
    def tearDown(self):
        """Clean up after each test method."""
        pass
        
    def test_haversine_distance(self):
        """Test the haversine distance calculation."""
        # Test data
        lat1, lon1 = 40.4168, -3.7038  # Madrid
        lat2, lon2 = 40.4170, -3.7040  # Nearby point
        
        # Call function under test
        distance = self.recommender.haversine(lat1, lon1, lat2, lon2)
        
        # Assertions
        self.assertGreater(distance, 0)
        self.assertLess(distance, 100)  # Should be very close
```

## Testing Recommendations Models

### SVD Recommender Tests

Key tests for the SVD-based recommender include:

- Testing distance calculations (haversine)
- Verifying matrix construction from user preferences
- Checking recommendation scoring logic
- Validating model fitting process
- Testing the end-to-end recommendation pipeline

Example test:

```python
def test_get_recommendations(self):
    """Test the recommendation generation pipeline."""
    # Prepare test data
    df = pd.DataFrame({
        'place_id': ['place1', 'place2', 'place3'],
        'name': ['Place 1', 'Place 2', 'Place 3'],
        'lat': [40.4168, 40.4268, 40.4368],
        'lng': [-3.7038, -3.7138, -3.7238],
        'types': ['museum,tourist_attraction', 'park', 'restaurant'],
        'rating': [4.5, 4.2, 4.0],
        'user_ratings_total': [1000, 500, 200]
    })
    
    predicted_ratings = {
        'museums': 4.5,
        'parks': 3.0,
        'restaurants': 2.0
    }
    
    # Mock the SVD model
    self.recommender.model = MagicMock()
    self.recommender.fit(df)
    
    # Call function under test
    recommendations = self.recommender.get_recommendations(
        df=df,
        user_lat=40.4168,
        user_lon=-3.7038,
        predicted_ratings=predicted_ratings,
        top_n=2
    )
    
    # Assertions
    self.assertEqual(len(recommendations), 2)
    self.assertEqual(recommendations[0]['place_id'], 'place1')  # Highest score should be first
```

### Autoencoder Recommender Tests

For the autoencoder-based recommender, we test:

- Preference vector preprocessing
- Model input/output handling
- Category mapping
- Recommendation generation with predicted preferences
- Distance and popularity adjustments

### Transfer Learning Recommender Tests

The transfer learning recommender tests focus on:

- Base model training and validation
- Embedding generation
- Domain transfer quality
- Place similarity calculations
- Preference-based recommendation filtering

### Ensemble Recommender Tests

Key tests for the ensemble recommender include:

- Score normalization across models
- Weight application
- Recommendation merging
- Diversity constraints
- Final ranking order

## Testing Helper Functions

We also have dedicated tests for utility functions:

- Data preprocessing functions
- Distance calculations
- Category mapping
- Route optimization algorithms
- Data validation functions

## Mocking External Dependencies

We use mocking to isolate units from their dependencies:

```python
@patch('tensorflow.keras.models.load_model')
def test_model_loading(self, mock_load_model):
    """Test model loading with mocked TensorFlow."""
    # Setup mock
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model
    
    # Call function that loads model
    recommender = AutoencoderRecommender(...)
    
    # Assertions
    mock_load_model.assert_called_once()
    self.assertEqual(recommender.model, mock_model)
```

## Parameterized Tests

For testing functions with multiple input combinations, we use parameterized tests:

```python
@unittest.parameterize([
    (40.4168, -3.7038, 40.4170, -3.7040, 30),  # Close points
    (40.4168, -3.7038, 41.3851, 2.1734, 500000),  # Madrid to Barcelona
])
def test_haversine_distances(self, lat1, lon1, lat2, lon2, expected_range):
    """Test distance calculations with various points."""
    distance = self.recommender.haversine(lat1, lon1, lat2, lon2)
    self.assertGreater(distance, 0)
    self.assertLess(distance, expected_range)
```

## Testing Exceptions

We test error conditions to ensure proper exception handling:

```python
def test_invalid_preference_score(self):
    """Test that invalid preference scores raise exceptions."""
    with self.assertRaises(ValueError):
        self.recommender.validate_preference(-1.0)  # Below minimum
        
    with self.assertRaises(ValueError):
        self.recommender.validate_preference(6.0)  # Above maximum
```

## Running Unit Tests

To run all unit tests:

```bash
python -m unittest discover -s reco/tests
```

To run a specific test file:

```bash
python -m unittest reco/tests/test_svd_recommender.py
```

To run a specific test case:

```bash
python -m unittest reco.tests.test_svd_recommender.TestSVDPlaceRecommender.test_haversine_distance
``` 