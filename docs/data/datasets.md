# Datasets

## Overview

Planwise combines multiple data sources to create a comprehensive recommendation system. Our data foundation includes user-place interactions, place metadata, and user reviews.

## Core Data Sources

### User-Place Ratings

**File**: `rating-California.csv`

This dataset contains user ratings of places on a scale of 1-5 stars:
- `user_id`: Unique identifier for each user
- `place_id`: Unique identifier for each place
- `rating`: Numerical rating from 1-5
- `timestamp`: When the rating was recorded

### Active User List

**File**: `filtered_users.csv`

To ensure quality recommendations, we focus on users with significant engagement:
- Lists Google Maps IDs for users who have contributed at least 20 ratings
- Ensures our models learn from genuinely engaged participants

### Place Metadata

**File**: `meta-California.json`

Rich metadata about each venue:
- Name
- Latitude/longitude coordinates
- Average community rating
- Total review count
- Array of Google Place types (e.g., ["art_gallery", "museum", ...])

### Madrid Places

**File**: `combined_places.csv`

Comprehensive dataset of Madrid venues:
- Place ID
- Name
- Location coordinates 
- Place categories
- Average ratings
- Review count
- Descriptions

### Review Sample

**File**: `review-California_10.json`

Sample of user reviews:
- Loaded in 1,000-line chunks for schema inspection
- Provides qualitative insights
- Useful for future feature ideas and sentiment analysis

## Derived Datasets

Through our preprocessing pipeline, we generate several intermediate datasets:

### Merged User-Place-Category Dataset

**File**: `users_ratings_categories.csv`

- Combines raw ratings with the active-user list
- Adds primary category information
- Columns: `user_id`, `place_id`, `rating`, `category1`

### User-Category Aggregation

**Files**: 
- `average_user_ratings_per_category.csv`
- `user_counts_per_category.csv`

These files provide aggregated metrics of how users rate different categories:
- Average rating per (user, category) combination
- Count of ratings per (user, category) combination

### Active-Category Filtered Dataset

**File**: `filtered_users_over_20_categories.csv`

- Contains IDs of users who have rated more than 20 distinct primary categories
- Used to filter the dataset to users with diverse experience

### Normalized Category Dataset

**File**: `final_users_over_20_categories.csv`

- Maps raw category codes to human-friendly buckets
- Example: "art_gallery" → "Art & Culture"
- This is the final cleaned dataset that feeds directly into all recommendation models

## Data Quality

All datasets undergo thorough validation:
- Removal of duplicate entries
- Handling of missing values
- Validation of rating ranges
- Geocoding verification for place coordinates 