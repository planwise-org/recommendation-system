# Planwise Recommendation System

## Overview

The Planwise Recommendation System is a hybrid ensemble recommender that leverages multiple machine learning techniques to provide personalized place recommendations in Madrid. By combining collaborative filtering, content-based filtering, and deep learning approaches, the system delivers recommendations that are both accurate and diverse.

## Features

- **Multi-model architecture**: Combines SVD, autoencoders, and transfer learning
- **Location-aware recommendations**: Considers user location and travel distance
- **Preference modeling**: Learns user preferences across multiple categories
- **Diversity optimization**: Prevents category dominance in recommendations
- **Cold-start handling**: Generates quality recommendations even with minimal user data

## Architecture

The recommendation system consists of four main recommender models:

1. **Autoencoder Recommender**: Deep learning model that reconstructs user preferences
2. **SVD Recommender**: Collaborative filtering using Singular Value Decomposition
3. **Transfer Recommender**: Cross-domain knowledge transfer from movie preferences to places
4. **Madrid Transfer Recommender**: Location-specific model using semantic embeddings
5. **Ensemble Recommender**: Meta-model that combines and weights outputs from other models

## Installation

```bash
# Clone the repository
git clone https://github.com/planwise-org/recommendation-system.git
cd recommendation-system/reco

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download model resources
python -m planwise.download_resources
```

## Usage

```python
from planwise.src.recommenders.ensemble_recommender import EnsembleRecommender
from planwise.src.utils.data_loader import load_places_data

# Load place data
places_df = load_places_data()

# Initialize recommender
recommender = EnsembleRecommender()
recommender.initialize_models(
    auto_model=...,  # Preloaded autoencoder model
    scaler=...,      # Feature scaler
    places_df=places_df,
    category_to_place_types=...  # Category mappings
)

# Get recommendations
user_preferences = {"museums": 5, "parks": 4, "restaurants": 3}
user_lat, user_lon = 40.4168, -3.7038  # Madrid center

recommendations = recommender.get_recommendations(
    user_lat=user_lat,
    user_lon=user_lon,
    user_prefs=user_preferences,
    predicted_ratings_dict=user_preferences,
    num_recs=10
)

# Display recommendations
for i, rec in enumerate(recommendations):
    print(f"{i+1}. {rec['name']} ({rec['category']}) - Score: {rec['score']:.2f}")
```

## Model Descriptions

### Autoencoder Recommender

Neural network that compresses and reconstructs user preference data, learning latent patterns to predict preferences for new places.

### SVD Recommender

Matrix factorization approach that decomposes the user-item interaction matrix to identify latent factors and predict ratings.

### Transfer Recommender

Uses a bidirectional variational autoencoder (BiVAE) pre-trained on MovieLens data, transferring knowledge from movie preferences to place recommendations.

### Madrid Transfer Recommender

Leverages sentence transformer embeddings to create a semantic space of Madrid places, matching user preferences with place descriptions.

### Ensemble Recommender

Combines predictions from all models, applying category diversity boosting, distance penalties, and novelty factors to create balanced recommendation lists.

## Testing

Run the test suite to ensure everything is working properly:

```bash
# Run all tests
python -m reco.tests.run_tests

# Run specific test
python -m reco.tests.test_ensemble_recommender
```

## Documentation

For detailed documentation, please visit our [GitHub Pages documentation site](https://planwise-org.github.io/recommendation-system/).
