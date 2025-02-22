import pandas as pd

# Load the dataset
file_path = "/Users/alexandrakhreiche/Desktop/code/planwise_chatbots/recommendation-system/api/combined_google_review_ratings.csv"  # Update with your actual path
df = pd.read_csv(file_path, low_memory=False)

# Drop the 'Unnamed: 25' column if it exists
if 'Unnamed: 25' in df.columns:
    df.drop(columns=['Unnamed: 25'], inplace=True)

# Merge 'Zoo' and 'zoos' into one column 'zoo'
if 'Zoo' in df.columns and 'zoos' in df.columns:
    df['zoo'] = df[['Zoo', 'zoos']].max(axis=1)
    df.drop(columns=['Zoo', 'zoos'], inplace=True)

# Merge 'Supermarket' and 'Supermarkets' into 'supermarket'
if 'Supermarket' in df.columns and 'Supermarkets' in df.columns:
    df['supermarket'] = df[['Supermarket', 'Supermarkets']].max(axis=1)
    df.drop(columns=['Supermarket', 'Supermarkets'], inplace=True)

# Merge 'Stores' and 'Stores and Shopping' into 'stores'
if 'Stores' in df.columns and 'Stores and Shopping' in df.columns:
    df['stores'] = df[['Stores', 'Stores and Shopping']].max(axis=1)
    df.drop(columns=['Stores', 'Stores and Shopping'], inplace=True)

# Merge 'parks' and 'Park' into 'parks'
if 'parks' in df.columns and 'Park' in df.columns:
    df['parks'] = df[['parks', 'Park']].max(axis=1)
    df.drop(columns=['Park'], inplace=True)

# Drop the 'other' column if it exists
if 'other' in df.columns:
    df.drop(columns=['other'], inplace=True)

# Save the cleaned dataset as new_dataset.csv
cleaned_file_path = "new_dataset.csv"
df.to_csv(cleaned_file_path, index=False)

print(f"✅ Dataset cleaned and saved as {cleaned_file_path}")
