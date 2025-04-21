# Autoencoder-Based Recommender

## Overview

Our Autoencoder-Based Recommender uses a neural network architecture to learn latent patterns in user preferences across different place categories. This model excels at uncovering hidden connections—like how a fondness for modern art might translate to recommendations for contemporary galleries.

## Technical Implementation

### Architecture

The autoencoder consists of two main components:

1. **Encoder**: Compresses user preference vectors into a lower-dimensional latent space
2. **Decoder**: Reconstructs the full user preference vector from the compressed representation

```
Input Layer (29 neurons) → Hidden Layer (16 neurons) → Latent Space (8 neurons) → Hidden Layer (16 neurons) → Output Layer (29 neurons)
```

### Input Representation

- Each user's preferences are represented as a vector of ratings across 29 categories
- Each element corresponds to a category (e.g., "restaurants", "museums", "parks")
- Values range from 0-5, indicating user preference strength for each category
- Unknown preferences are initially set to 0

### Training Process

1. **Data Preparation**:
   - User preference vectors are normalized to [0,1] range
   - Known preferences are maintained for reconstruction targets
   - Missing preferences are masked during loss calculation

2. **Optimization**:
   - We use Mean Squared Error loss function
   - Adam optimizer with learning rate of 0.001
   - Early stopping to prevent overfitting
   - Dropout layers (rate=0.2) for regularization

3. **Denoising**:
   - Random noise is added to inputs during training
   - Forces the autoencoder to learn robust feature representations
   - Improves generalization to new users

### Recommendation Generation

The recommendation process follows these steps:

1. **Preference Prediction**:
   - Take user's explicit preferences as input
   - Pass through the trained autoencoder
   - The output contains predicted ratings for categories the user hasn't explicitly rated

2. **Place Scoring**:
   - For each venue, take the predicted score for its primary category
   - Scale by the venue's average community rating
   - Apply a boost based on total review count
   - Apply a small penalty proportional to distance (via Haversine formula)

3. **Ranking**:
   - Sort places by their final scores
   - Return top-N recommendations

## Implementation in Code

The AutoencoderRecommender class implements this model:

```python
class AutoencoderRecommender:
    def __init__(self, auto_model, scaler, places_df, categories_list, category_mappings):
        self.model = auto_model  # Pretrained TensorFlow autoencoder
        self.scaler = scaler     # For normalizing preference vectors
        self.places_df = places_df
        self.categories = categories_list
        self.category_to_place_types = category_mappings
        
    def get_recommendations(self, user_lat, user_lon, user_prefs, provided_mask, num_recs=5):
        # Implementation of recommendation algorithm
        # ...
```

## Advantages

- **Handles Sparse Input**: Produces quality recommendations even when users have only rated a few categories
- **Discovers Latent Patterns**: Captures non-obvious relationships between preferences
- **Personalization**: Tailors recommendations to individual preference profiles
- **Cold-Start Solution**: Can generate recommendations with minimal user input

## Limitations and Future Improvements

- **Training Data Requirements**: Needs substantial user-category ratings for training
- **Interpretability**: Latent factors are less interpretable than explicit features
- **Future Work**: We plan to implement a variational autoencoder (VAE) to better model the uncertainty in user preferences 