# Testing the Recommendation System

This directory contains unit tests for the recommendation system components. The tests validate the functionality of:

1. SVD-based Recommender
2. Transfer Learning Recommender
3. Autoencoder Recommender
4. Ensemble Recommender
5. Helper Functions

## Running Tests

### Important Note on Test Location

Before anything, go to the `reco` folder to run this test suite. The folder is the parent directory of this `tests`folder, and this is necessary for correct module import and path resolution.

### Running All Tests

To run all tests at once, use the `run_tests.py` script:

```bash
python /tests/run_tests.py
```


### Running Individual Test Files

You can also run individual test files directly:

```bash
python -m tests.test_svd_recommender
python -m tests.test_transfer_recommender
python -m tests.test_autoencoder_recommender
python -m tests.test_ensemble_recommender
python -m tests.test_meta_learner_ensemble
```

### Running Specific Test Cases

To run specific test cases, you can use the unittest framework directly:

```bash
python -m unittest tests.test_svd_recommender.TestSVDPlaceRecommender.test_haversine_distance
```

## Test Coverage

To check test coverage, you can use the `coverage` package:

```bash
# Install coverage if not already installed
pip install coverage

# Run tests with coverage
coverage run reco/tests/run_tests.py

# Generate a coverage report
coverage report -m

# Generate an HTML coverage report
coverage html
```

## Adding New Tests

When adding new functionality to the recommendation system, please also add corresponding test cases:

1. Create a new test file in this directory if testing a new component
2. Follow the naming convention: `test_*.py`
3. Use the unittest framework and the established patterns

## Dependencies

The tests require the following packages:
- unittest (standard library)
- mock (part of standard library as unittest.mock)
- numpy
- pandas
- scikit-learn
- surprise (for SVD)
- tensorflow (for autoencoder)

Make sure all dependencies are installed before running the tests. 