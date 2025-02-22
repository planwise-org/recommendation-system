
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os

# -----------------------------
# 1. EDA on rating-California.csv
# -----------------------------
def eda_ratings():
    print("Loading rating-California.csv for EDA...")
    df_rating = pd.read_csv('rating-California.csv')
    
    # Display basic information
    print("First few rows of the ratings DataFrame:")
    print(df_rating.head())
    print("\nBasic statistics:")
    print(df_rating.describe())
    print("\nMissing values:")
    print(df_rating.isnull().sum())
    
    # Plot distribution of ratings
    plt.figure(figsize=(10, 6))
    sns.histplot(df_rating['rating'], bins=20, kde=True)
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.show()
    
    # Plot relationships between rating and other numerical features
    numerical_features = df_rating.select_dtypes(include=['float64', 'int64']).columns
    for feature in numerical_features:
        if feature != 'rating':
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=df_rating[feature], y=df_rating['rating'])
            plt.title(f'Relationship between {feature} and Rating')
            plt.xlabel(feature)
            plt.ylabel('Rating')
            plt.show()

# -----------------------------
# 2. Preview a chunk from review-California_10.json
# -----------------------------
def review_chunk_preview():
    print("\nLoading a chunk from review-California_10.json...")
    chunk_size = 1000
    json_reader = pd.read_json('review-California_10.json', lines=True, chunksize=chunk_size)
    first_chunk = next(json_reader)
    print("First few rows of the review chunk:")
    print(first_chunk.head())
    print("\nColumns in the review chunk:")
    print(first_chunk.columns)

# -----------------------------
# 3. Merge filtered_users.csv with meta-California.json to add categories
# -----------------------------
def merge_users_meta():
    print("\nMerging filtered_users.csv with meta-California.json...")
    df_filtered = pd.read_csv('filtered_users.csv')
    print("Sample filtered users data:")
    print(df_filtered.head())
    gmap_ids = set(df_filtered['gmap_id'].unique())
    meta_chunks = []
    chunk_size = 10000  
    meta_reader = pd.read_json('meta-California.json', lines=True, chunksize=chunk_size)
    for chunk in meta_reader:
        filtered_chunk = chunk[chunk['gmap_id'].isin(gmap_ids)]
        if not filtered_chunk.empty:
            meta_chunks.append(filtered_chunk[['gmap_id', 'category']])
    if meta_chunks:
        df_meta = pd.concat(meta_chunks, ignore_index=True)
    else:
        print("No matching meta data found for the filtered users.")
        return None
    
    print("Sample meta data:")
    print(df_meta.head())
    
    def extract_first_category(cat):
        if isinstance(cat, list):
            return cat[0] if len(cat) > 0 else None
    
        try:
            parsed = ast.literal_eval(cat)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed[0]
            else:
                return cat
        except:
            return cat
    
    
    df_meta['category'] = df_meta['category'].apply(extract_first_category)
    
    df_merged = pd.merge(df_filtered, df_meta, on='gmap_id', how='left')
    print("Sample merged data (users with categories):")
    print(df_merged.head())
    
    df_merged.to_csv('users_ratings_categories.csv', index=False)
    print("Saved merged data to users_ratings_categories.csv")
    
    return df_merged

# -----------------------------
# 4. Compute average rating per user per category and count unique users/categories
# -----------------------------
def average_ratings_and_counts(df_merged):
    print("\nComputing average rating per user per category...")
    df_avg = df_merged.groupby(['user_id', 'category'], as_index=False)['rating'].mean()
    df_avg.rename(columns={'rating': 'average_rating'}, inplace=True)
    df_avg.to_csv('average_user_ratings_per_category.csv', index=False)
    print("Saved average ratings data to average_user_ratings_per_category.csv")
    
    # Count and print the number of unique users and categories
    num_users = df_avg['user_id'].nunique()
    num_categories = df_avg['category'].nunique()
    print(f"Number of individual users: {num_users}")
    print(f"Number of categories: {num_categories}")
    
    return df_avg

# -----------------------------
# 5. Save all unique categories into categories.csv
# -----------------------------
def save_unique_categories(df_avg):
    print("\nSaving unique categories to categories.csv...")
    unique_categories = pd.DataFrame(df_avg['category'].unique(), columns=['category'])
    unique_categories.to_csv('categories.csv', index=False)
    print("Saved unique categories to categories.csv")
    return unique_categories

# -----------------------------
# 6. Filter dataset for users who have rated over 20 categories
# -----------------------------
def filter_users_over_20_categories(df_avg):
    print("\nFiltering users who have rated over 20 different categories...")
    user_category_counts = df_avg.groupby('user_id')['category'].nunique().reset_index(name='num_categories')
    # Count how many users have rated over 10 categories (if needed)
    num_over_10 = (user_category_counts['num_categories'] > 10).sum()
    print(f"Number of users who have rated over 10 categories: {num_over_10}")
    
    # Now, filter for users with more than 20 categories
    users_over_20 = user_category_counts[user_category_counts['num_categories'] > 20]
    df_filtered_over_20 = df_avg[df_avg['user_id'].isin(users_over_20['user_id'])]
    df_filtered_over_20.to_csv('filtered_users_over_20_categories.csv', index=False)
    print("Saved filtered users (over 20 categories) to filtered_users_over_20_categories.csv")
    return df_filtered_over_20


# -----------------------------
# 7. Map categories to new names using refined_category_mapping_from_csv.py
# -----------------------------
def map_categories(df_filtered_over_20):
    print("\nMapping categories to new names using refined_category_mapping_from_csv.py...")
    try:
        # The file refined_category_mapping_from_csv.py should define a dictionary called category_mapping
        from refined_category_mapping_from_csv import category_mapping
    except ImportError:
        print("Could not import category_mapping from refined_category_mapping_from_csv.py. Proceeding without mapping.")
        category_mapping = {}
    
    # Map the categories using the provided mapping (if available)
    df_filtered_over_20['refined_category'] = df_filtered_over_20['category'].map(lambda x: category_mapping.get(x, x))
    df_filtered_over_20.to_csv('final_users_over_20_categories.csv', index=False)
    print("Saved final mapped data to final_users_over_20_categories.csv")
    return df_filtered_over_20

# -----------------------------
# Main pipeline execution
# -----------------------------
def main():
    # Step 1: EDA on ratings
    eda_ratings()
    
    # Step 2: Preview reviews chunk
    review_chunk_preview()
    
    # Step 3: Merge filtered users with meta data to add categories
    df_merged = merge_users_meta()
    if df_merged is None:
        print("Merging failed; stopping execution.")
        return
    
    # Step 4: Compute average ratings per user-category and count unique users/categories
    df_avg = average_ratings_and_counts(df_merged)
    
    # Step 5: Save unique categories
    save_unique_categories(df_avg)
    
    # Step 6: Filter dataset for users with over 20 rated categories
    df_filtered_over_20 = filter_users_over_20_categories(df_avg)
    
    # Step 7: Filter out irrelevant categories from the categories file
    filter_irrelevant_categories()
    
    # Step 8: Map categories to refined names based on external mapping
    map_categories(df_filtered_over_20)

if __name__ == "__main__":
    main()