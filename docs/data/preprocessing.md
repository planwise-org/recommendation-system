# Data Preprocessing

## Overview

To build recommendation models on a solid foundation, we designed a repeatable pipeline that ingests, cleans, and inspects our raw data before any modeling begins. This ensures consistent, high-quality data for all recommendation algorithms.

## Preprocessing Pipeline

All transformation logic is encapsulated in `data/userspipeline.py`, which runs these deterministic steps:

### 1. Data Loading

The pipeline begins by loading the raw data files:
- User-place ratings from `rating-California.csv`
- Active user list from `filtered_users.csv`
- Place metadata from `meta-California.json`

### 2. Join & Filter

- Inner-join raw ratings with the active-user list
- Discard interactions from less-active accounts (fewer than 20 ratings)
- Flatten the nested JSON metadata so that every attribute becomes a regular DataFrame column

### 3. Primary Category Extraction

- Parse each place's types list (e.g., ["art_gallery", "museum", "tourist_attraction"])
- Extract the first element into a new column, `category1`
- This primary tag drives our downstream grouping and preference modeling

### 4. Intermediate Persistence

- Save the merged DataFrame (columns: `user_id`, `place_id`, `rating`, `category1`) as `users_ratings_categories.csv`
- This intermediate file provides transparency and enables reuse in different model pipelines

### 5. Per-User, Per-Category Aggregation

- Group by (`user_id`, `category1`) to compute each user's average rating per category
- Export these summaries to `average_user_ratings_per_category.csv`
- Save count information to `user_counts_per_category.csv`

### 6. Active-Category Filtering

- Identify users who have rated more than 20 distinct primary categories
- Write their IDs to `filtered_users_over_20_categories.csv`
- This ensures we model users with diverse experiences across category types

### 7. Category Normalization

- Apply the lookup in `refined_category_mapping_from_csv.py` to map raw codes to human-friendly buckets
  - Example: "art_gallery" → "Art & Culture"
  - Example: "restaurant" → "Food & Drink"
- Save the final cleaned dataset as `final_users_over_20_categories.csv`

## Madrid Data Processing

For Madrid-specific data:

1. **Collection**: We gathered place data for Madrid using the Google Places API
2. **Cleaning**: Removed duplicates and places with missing critical information
3. **Categorization**: Mapped Google place types to our normalized category system
4. **Embedding**: Generated text embeddings for places to support our Madrid Embedding Recommender
5. **Storage**: Saved the processed data to `combined_places.csv` and the embeddings to `madrid_place_embeddings.npz`

## Data Validation

Throughout the pipeline, we implement validation checks:

- **Schema Validation**: Ensure all required columns are present
- **Value Range Checks**: Validate that ratings fall within the expected 1-5 scale
- **Null Detection**: Identify and handle missing values
- **Integrity Checks**: Verify that foreign keys exist in the related tables
- **Geocoding Validation**: Ensure coordinates fall within expected bounds

By automating these steps in a single, transparent pipeline, we guarantee that every dataset entering model training has been consistently cleaned and comprehensively characterized—laying a solid foundation for all recommendation algorithms. 