# Transfer Learning Recommender

## Overview

Our Transfer Learning Recommender adapts a pretrained bidirectional Variational Autoencoder (BiVAE) to enable cross-domain recommendation. This innovative approach leverages patterns learned from a rich source domain (MovieLens dataset) to make quality recommendations in our target domain (places in Madrid) with limited data.

## Technical Implementation

### BiVAE Architecture

The Bidirectional Variational Autoencoder consists of:

1. **Encoder Network**: Maps input ratings to a probabilistic latent space
2. **Decoder Network**: Reconstructs ratings from latent representations
3. **Bidirectional Mapping**: Enables transfer between source and target domains

```
User/Item Inputs → Encoder (μ, σ) → Latent Space (z) → Decoder → Reconstructed Preferences
```

### Cross-Domain Transfer

We bridge the two domains through these key steps:

1. **Pretraining**: The BiVAE learns genre embeddings on the MovieLens 100k dataset, which contains rich user-movie rating data

2. **Category Mapping**: We establish a semantic alignment between:
   - Movie genres (source domain)
   - Place categories (target domain)

   Examples:
   - "documentary" (movie) → "museum" (place)
   - "nature" (movie) → "park" (place)
   - "food" (movie) → "café" (place)

3. **Embedding Assignment**:
   - Each venue inherits its genre's latent vector from the movie domain
   - Each user's embedding is computed as the weighted average of embeddings for their top-rated categories

### Recommendation Process

The recommendation pipeline follows these steps:

1. **User Embedding Creation**: 
   - From user category preferences, create a weighted embedding vector
   - Higher weights are given to categories with higher preference ratings

2. **Similarity Scoring**:
   - Compute cosine similarity between the user embedding and all place embeddings
   - Apply a category match boost if the predicted category exactly matches the place's primary tag
   - Apply distance and popularity modifiers

3. **Ranking**:
   - Sort places by their final similarity scores
   - Return the top-N recommendations

## Implementation in Code

The TransferRecommender class implements this approach:

```python
class TransferRecommender:
    def __init__(self):
        self.base_model = None
        self.genre_embeddings = {}
        self.place_embeddings = {}
        
    def train_base_model(self):
        # Train or load pretrained BiVAE on MovieLens
        # Save genre embeddings to self.genre_embeddings
        # ...
        
    def transfer_to_places(self, places_df):
        # Map place categories to movie genres
        # Assign embeddings to places
        # ...
        
    def get_recommendations(self, user_preferences, user_lat, user_lon, places_df, top_n=5):
        # Generate user embedding from preferences
        # Compute similarities and recommend places
        # ...
```

## Advantages

- **Cold-Start Solution**: Excels at generating recommendations with minimal user data
- **Knowledge Transfer**: Leverages rich patterns from source domain
- **Semantic Understanding**: Captures deep, non-linear relationships between preferences
- **Efficient**: Requires less target domain data than training models from scratch

## Limitations and Future Improvements

- **Domain Alignment**: Performance depends on quality of cross-domain mapping
- **User Adaptation**: Less personalized for users with preferences that differ significantly from source domain patterns
- **Future Work**:
  - Fine-tune the mapping between domains
  - Implement adaptive weighting for better personalization
  - Explore multi-domain transfer with additional data sources 