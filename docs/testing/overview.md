# Testing Overview

## Introduction

Testing is a crucial part of the Planwise development process. We employ a comprehensive testing strategy to ensure the reliability, functionality, and performance of our recommendation system components. This document provides an overview of our testing approach and how to run different types of tests.

## Testing Components

Our testing suite validates the functionality of several key components:

1. **SVD-based Recommender**: Tests collaborative filtering recommendations
2. **Transfer Learning Recommender**: Tests cross-domain recommendation capabilities
3. **Autoencoder Recommender**: Tests neural network-based recommendations
4. **Ensemble Recommender**: Tests the combination of multiple recommendation strategies
5. **API Endpoints**: Tests FastAPI routes and services
6. **Helper Functions**: Tests utility functions used across the system

## Test Directory Structure

```
recommendation-system/
├── reco/
│   ├── tests/
│   │   ├── run_tests.py           # Script to run all tests
│   │   ├── test_svd_recommender.py
│   │   ├── test_transfer_recommender.py
│   │   ├── test_autoencoder_recommender.py
│   │   ├── test_ensemble_recommender.py
│   │   └── test_utils.py
├── api/
│   ├── tests/
│   │   ├── conftest.py            # Test fixtures and configuration
│   │   ├── test_auth.py           # Authentication tests
│   │   ├── test_users.py          # User API tests
│   │   ├── test_places.py         # Places API tests
│   │   ├── test_reviews.py        # Reviews API tests
│   │   ├── test_preferences.py    # Preferences API tests
│   │   └── test_recommendations.py # Recommendation API tests
```

## Running Tests

### Running All Tests

To run all tests at once, use the `run_tests.py` script:

```bash
python reco/tests/run_tests.py
```

This will discover and run all test files in the directory.

### Running Individual Test Files

You can also run individual test files directly:

```bash
python -m reco.tests.test_svd_recommender
python -m reco.tests.test_transfer_recommender
python -m reco.tests.test_autoencoder_recommender
python -m reco.tests.test_ensemble_recommender
```

### Running API Tests

To run API tests:

```bash
cd api
pytest
```

### Running Specific Test Cases

To run specific test cases, you can use the unittest framework directly:

```bash
python -m unittest reco.tests.test_svd_recommender.TestSVDPlaceRecommender.test_haversine_distance
```

Or with pytest:

```bash
pytest api/tests/test_auth.py::test_login_success
```

## Test Coverage

To check test coverage, we use the `coverage` package:

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

For API tests with coverage:

```bash
cd api
pytest --cov=src --cov-report=term-missing
```

## Test Types

### Unit Tests

Unit tests verify individual components in isolation, focusing on specific functions and methods. They are designed to be fast and to pinpoint issues precisely.

### Integration Tests

Integration tests verify that components work together correctly. These tests often involve multiple modules and may interact with external dependencies (like databases).

### Functional Tests

Functional tests verify that the system meets business requirements from an end-user perspective. They test complete features and workflows.

### Performance Tests

Performance tests measure system responsiveness and stability under various conditions, including load testing and stress testing.

## Test Dependencies

The tests require the following packages:
- unittest (standard library)
- mock (part of standard library as unittest.mock)
- pytest (for API tests)
- pytest-cov (for coverage reporting)
- httpx (for async API testing)
- numpy
- pandas
- scikit-learn
- surprise (for SVD)
- tensorflow (for autoencoder)

Make sure all dependencies are installed before running the tests:

```bash
pip install -r requirements-dev.txt
```

## Adding New Tests

When adding new functionality to the recommendation system, please also add corresponding test cases:

1. Create a new test file in the appropriate tests directory if testing a new component
2. Follow the naming convention: `test_*.py`
3. Use the unittest framework and the established patterns
4. Ensure adequate test coverage for new code

## Continuous Integration

All tests are automatically run on our CI/CD pipeline using GitHub Actions. Tests must pass before any pull request can be merged.

## Mocking

For tests that would otherwise require external dependencies, we use mocking:

```python
from unittest.mock import patch, MagicMock

@patch('path.to.dependency', return_value=MagicMock())
def test_function_with_dependency(mock_dependency):
    # Test implementation
    assert mock_dependency.called
```

## Testing Best Practices

1. **Test Independence**: Each test should be independent and not rely on the state from previous tests
2. **Test Coverage**: Aim for at least 80% test coverage
3. **Fast Tests**: Tests should run quickly to encourage frequent testing
4. **Descriptive Test Names**: Test names should clearly describe what they're testing
5. **Assertions**: Use specific assertions that provide helpful failure messages 