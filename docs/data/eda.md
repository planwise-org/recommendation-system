# Exploratory Data Analysis

## Overview

We generate all charts and summaries at runtime to keep insights current. Our EDA process helps us understand data distributions, identify patterns, and detect anomalies before feeding data into recommendation models.

## EDA Components

### Summary Statistics & Missing-Value Report

The function `eda_ratings()` performs several key operations:
- Loads the ratings DataFrame
- Displays the first few rows for initial inspection
- Generates descriptive statistics (`df.describe()`)
- Reports null counts across all columns (`df.isnull().sum()`)

Example output for ratings distribution:

```
count    1234567.000000
mean         3.752984
std          1.254637
min          1.000000
25%          3.000000
50%          4.000000
75%          5.000000
max          5.000000
```

### Rating Distribution Visualization

Within `eda_ratings()`, we generate:
- 20-bin histogram of ratings
- KDE (Kernel Density Estimation) overlay
- This visualization helps identify the overall shape of user feedback and potential biases

### User Activity Analysis

We analyze user engagement patterns:
- Distribution of ratings per user
- Histogram of number of places rated by each user
- Identification of super-users vs. casual users

### Place Popularity Analysis

For places, we examine:
- Distribution of ratings received per place
- Average ratings across places
- Most and least rated categories

### Geographic Distribution

Using the place metadata:
- We map locations to visualize geographic clustering
- Analyze rating patterns by region
- Identify areas with sparse coverage

### Category Analysis

We perform detailed analysis of categories:
- Distribution of places across categories
- Average ratings by category
- User preference patterns across categories

### Scatter Plots of Numeric Features

To detect anomalies or unintended biases:
- `eda_ratings()` iterates over each numeric column (e.g., `user_id`, `place_id`, `timestamp`)
- Plots each against `rating` to visualize relationships
- Identifies outliers or unusual patterns

### Review Schema Inspection

The helper `review_chunk_preview()`:
- Reads and previews the first 1,000 lines of the JSON reviews file
- Displays the DataFrame head and column list
- Verifies structure without exposing text content

## Key Insights

Our EDA revealed several important patterns:

1. **Rating Distribution**: A positive skew with most ratings in the 4-5 range
2. **Category Preferences**: Museums and parks have higher average ratings than restaurants and bars
3. **Geographic Patterns**: Central Madrid has higher venue density but more rating variability
4. **User Segments**: Clear distinction between locals (rating diverse categories) and tourists (rating primarily attractions)
5. **Temporal Patterns**: Seasonal variations in ratings for outdoor vs. indoor venues

By automating these steps in a single, transparent pipeline, we guarantee that every dataset entering model training has been consistently cleaned and comprehensively characterized—laying a solid foundation for all recommendation algorithms. 