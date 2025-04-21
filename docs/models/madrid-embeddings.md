# Madrid Embedding Recommender

## Overview

The Madrid Embedding Recommender enables fast, cold-start recommendations specifically tailored for Madrid without requiring local user reviews. This lightweight model uses text embeddings to match user preferences with relevant places, making it particularly valuable for new users or visitors to the city.

## Technical Implementation

### Text Embedding Approach

Our approach leverages modern natural language processing techniques:

1. **Preference Encoding**: User preferences are converted into a pseudo-document format:
   - Categories are repeated according to their preference score
   - Example: A user with high preference for parks (4.5) and cafes (3.5) is represented as "parks parks parks parks cafes cafes cafes"
   - This weighted text representation captures preference strength

2. **Embedding Model**: We use the all-MiniLM-L6-v2 SentenceTransformer model:
   - Produces 384-dimensional dense vector representations
   - Pre-trained on diverse text datasets
   - Optimized for semantic similarity tasks

3. **Place Embedding**: Each venue is represented as a text description including:
   - Place name
   - Category information
   - Key features and attributes
   - These are embedded into the same vector space as user preferences

### Recommendation Pipeline

The recommendation process follows these steps:

1. **User Embedding Creation**:
   - Extract category preferences from user input (via sliders or chatbot)
   - Generate the weighted category text representation
   - Compute the embedding vector using SentenceTransformer

2. **Similarity Computation**:
   - Calculate cosine similarity between user embedding and all precomputed place embeddings
   - Higher similarity indicates better semantic match between user preferences and place attributes

3. **Filtering & Diversity**:
   - Apply distance filter (3 km radius from user's location)
   - Group results by category
   - Apply a diversity constraint: maximum of two recommendations per category
   - This ensures varied recommendations rather than many similar places

4. **Final Ranking**:
   - Sort by similarity score
   - Apply popularity and distance adjustments
   - Return top-N recommendations

## Implementation in Code

The MadridTransferRecommender class implements this approach:

```python
class MadridTransferRecommender:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2', embedding_path='models/madrid_place_embeddings.npz'):
        self.model = SentenceTransformer(embedding_model_name)
        self.place_embeddings = self._load_embeddings(embedding_path)
        self.place_metadata = self._load_place_metadata()
        
    def _load_embeddings(self, path):
        # Load precomputed place embeddings
        # ...
        
    def _create_user_embedding(self, user_prefs):
        # Convert user preferences to text and embed
        # ...
        
    def get_recommendations(self, user_lat, user_lon, user_prefs, num_recs=5):
        # Generate recommendations based on embedding similarity
        # ...
```

## Advantages

- **No User History Required**: Perfect for new users with no previous interactions
- **Fast Inference**: Pre-computed embeddings enable real-time recommendations
- **Language Understanding**: Captures semantic relationships between preferences and places
- **Diversity**: Built-in constraints ensure varied recommendations
- **Location Awareness**: Geographic filtering ensures relevant local suggestions

## Limitations and Future Improvements

- **Limited Personalization**: Less tailored to individual user history than collaborative filtering approaches
- **Vocabulary Dependence**: Performance affected by how places and preferences are described
- **Future Work**:
  - Fine-tune embeddings on domain-specific data
  - Integrate user feedback to improve embeddings over time
  - Experiment with more sophisticated text representations 