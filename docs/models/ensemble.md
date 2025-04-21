# Ensemble Recommender

## Overview

The Ensemble Recommender combines the strengths of all four individual models—Autoencoder, SVD, Transfer Learning, and Madrid Embeddings—to produce more robust, diverse, and accurate recommendations. By intelligently blending different recommendation strategies, we overcome the limitations of any single approach and deliver superior user experiences.

## Technical Implementation

### Ensemble Architecture

Our ensemble combines model outputs using a weighted fusion approach:

1. **Individual Model Predictions**: Each model generates its own set of recommendations with scores
2. **Score Normalization**: Scores from each model are scaled to a common [0,1] range
3. **Weighted Combination**: Normalized scores are combined using a configurable weighting scheme
4. **Diversity & Novelty Adjustments**: Additional logic applies diversity constraints and novelty boosts

### Model Weights

We assign the following default weights, tuned through iterative testing:

- **20%** Autoencoder (general user preferences)
- **20%** SVD (popular, highly-rated places)
- **30%** California Transfer Learning (cross-regional semantic matching)
- **30%** Madrid Embedding Recommender (local real-time personalization)

These weights balance the strengths of each model:
- The autoencoder captures complex preference patterns
- SVD identifies generally popular venues
- Transfer learning leverages cross-domain knowledge
- Madrid embeddings provide location-specific relevance

### Scoring Enhancements

To mitigate redundancy and improve recommendation quality, we incorporate:

1. **Distance Penalty**: A function that softly prefers nearby venues while allowing exceptional matches farther out:
   ```python
   distance_penalty = max(0, 1 - (distance / max_distance))
   ```

2. **Category Diversity**: A constraint that limits recommendations to maximum of 2 venues per category:
   ```python
   recommendations_by_category = defaultdict(list)
   for place in sorted_candidates:
       if len(recommendations_by_category[place['category']]) < 2:
           recommendations_by_category[place['category']].append(place)
   ```

3. **Novelty Boost**: A small bonus for lesser-known venues to reduce overrepresentation of popular tourist spots:
   ```python
   novelty_boost = 0.1 * (1 - min(1.0, math.log(place['user_ratings_total'] + 1) / 10))
   ```

### Recommendation Generation

The recommendation process follows these steps:

1. **Collect Predictions**: Get top-N recommendations from each individual model
2. **Normalize Scores**: Scale each model's scores to [0,1] range
3. **Apply Weights**: Multiply normalized scores by model weights
4. **Combine Results**: Merge candidates, summing weighted scores for duplicates
5. **Apply Enhancements**: Add distance penalties, diversity constraints, and novelty boosts
6. **Final Ranking**: Sort by adjusted scores and return top recommendations

## Implementation in Code

The EnsembleRecommender class implements this approach:

```python
class EnsembleRecommender:
    def __init__(self):
        self.models = {}
        self.weights = {
            'autoencoder': 0.2,
            'svd': 0.2,
            'transfer': 0.3,
            'madrid_embedding': 0.3
        }
        
    def initialize_models(self, **kwargs):
        # Initialize individual models with provided parameters
        # ...
        
    def normalize_scores(self, recommendations):
        # Scale scores to [0,1] range
        # ...
        
    def get_recommendations(self, user_lat, user_lon, user_prefs, 
                          predicted_ratings_dict, num_recs=5, **kwargs):
        # Get predictions from each model
        # Normalize and combine scores
        # Apply enhancements and return final recommendations
        # ...
```

## Advantages

- **Robustness**: Less vulnerable to weaknesses of any single algorithm
- **Balanced Recommendations**: Combines popularity, personalization, and discovery
- **Adaptability**: Weights can be adjusted based on performance or business goals
- **Diversity**: Built-in mechanisms ensure varied recommendations
- **Scalability**: New models can be added to the ensemble over time

## Limitations and Future Improvements

- **Complexity**: More moving parts than single-algorithm approaches
- **Tuning Needed**: Finding optimal weights requires experimentation
- **Computational Cost**: Requires running multiple models for each recommendation request

- **Future Work**:
  - Implement adaptive weights based on user context
  - Add A/B testing to automatically optimize model weights
  - Explore stacked ensembles with a meta-learner 