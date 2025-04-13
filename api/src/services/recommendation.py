from typing import List, Dict
from sqlmodel import Session, select
from ..models import User, Place, Review, Recommendation, RecommendationAlgorithm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import random

def create_user_place_matrix(db: Session) -> tuple:
    """Create a user-place matrix from reviews."""
    # Get all reviews
    reviews = db.exec(select(Review)).all()
    
    # Get unique users and places
    users = db.exec(select(User)).all()
    places = db.exec(select(Place)).all()
    
    # Create user and place dictionaries for indexing
    user_dict = {user.id: idx for idx, user in enumerate(users)}
    place_dict = {place.id: idx for idx, place in enumerate(places)}
    
    # Initialize the matrix with zeros
    matrix = np.zeros((len(users), len(places)))
    
    # Fill the matrix with ratings
    for review in reviews:
        user_idx = user_dict[review.user_id]
        place_idx = place_dict[review.place_id]
        matrix[user_idx, place_idx] = review.rating
    
    return matrix, user_dict, place_dict

def autoencoder_recommendations(
    db: Session,
    user_id: int,
    n_recommendations: int = 5
) -> List[Dict]:
    """Generate recommendations using a simple approach."""
    # Get user
    user = db.get(User, user_id)
    if not user:
        raise ValueError(f"User {user_id} not found")
    
    # Get all places
    places = db.exec(select(Place)).all()
    if not places:
        return []
    
    # Get user's reviewed places
    user_reviews = db.exec(select(Review).where(Review.user_id == user_id)).all()
    reviewed_place_ids = {review.place_id for review in user_reviews}
    
    # Filter out places the user has already reviewed
    available_places = [place for place in places if place.id not in reviewed_place_ids]
    if not available_places:
        return []
    
    # Get average ratings for each place
    place_ratings = {}
    for place in available_places:
        place_reviews = db.exec(select(Review).where(Review.place_id == place.id)).all()
        if place_reviews:
            avg_rating = sum(review.rating for review in place_reviews) / len(place_reviews)
            place_ratings[place.id] = avg_rating / 5.0  # Normalize to 0-1 range
        else:
            place_ratings[place.id] = 0.5  # Default score for places with no reviews
    
    # Sort places by rating
    sorted_places = sorted(available_places, key=lambda p: place_ratings[p.id], reverse=True)
    
    # Take top N places
    n_recommendations = min(n_recommendations, len(sorted_places))
    selected_places = sorted_places[:n_recommendations]
    
    # Create recommendations
    recommendations = []
    for place in selected_places:
        recommendations.append({
            "user_id": user_id,
            "place_id": place.id,
            "algorithm": RecommendationAlgorithm.AUTOENCODER.value,
            "score": place_ratings[place.id]
        })
    
    return recommendations

def svd_recommendations(
    db: Session,
    user_id: int,
    n_recommendations: int = 5
) -> List[Dict]:
    """Generate recommendations using Singular Value Decomposition."""
    # Get user
    user = db.get(User, user_id)
    if not user:
        raise ValueError(f"User {user_id} not found")
    
    # Create user-place matrix
    matrix, user_dict, place_dict = create_user_place_matrix(db)
    
    # Get user index
    user_idx = user_dict[user.id]
    
    # Perform SVD
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
    
    # Choose number of components (can be tuned)
    n_components = min(len(s), 20)
    
    # Reconstruct matrix with reduced dimensions
    matrix_reconstructed = U[:, :n_components] @ np.diag(s[:n_components]) @ Vt[:n_components, :]
    
    # Get user's predicted ratings
    predicted_ratings = matrix_reconstructed[user_idx]
    
    # Get top N recommendations for unrated places
    user_ratings = matrix[user_idx]
    unrated_mask = user_ratings == 0
    top_indices = np.argsort(predicted_ratings * unrated_mask)[-n_recommendations:][::-1]
    
    # Convert place indices back to place IDs
    place_id_dict = {idx: place_id for place_id, idx in place_dict.items()}
    recommendations = []
    for idx in top_indices:
        place_id = place_id_dict[idx]
        recommendations.append({
            "user_id": user_id,
            "place_id": place_id,
            "algorithm": RecommendationAlgorithm.SVD.value,
            "score": float(predicted_ratings[idx])
        })
    
    return recommendations

def transfer_learning_recommendations(
    db: Session,
    user_id: int,
    n_recommendations: int = 5
) -> List[Dict]:
    """Generate recommendations using transfer learning approach."""
    # Get user
    user = db.get(User, user_id)
    if not user:
        raise ValueError(f"User {user_id} not found")
    
    # Create user-place matrix
    matrix, user_dict, place_dict = create_user_place_matrix(db)
    
    # Get user index
    user_idx = user_dict[user.id]
    
    # Get user's reviews
    user_reviews = db.exec(select(Review).where(Review.user_id == user_id)).all()
    
    # Create feature vectors for places
    places = db.exec(select(Place)).all()
    place_features = []
    for place in places:
        # Get place's average rating
        place_reviews = db.exec(select(Review).where(Review.place_id == place.id)).all()
        avg_rating = np.mean([r.rating for r in place_reviews]) if place_reviews else 0
        
        features = [
            avg_rating,
            len(place_reviews),  # number of reviews
            1 if place.place_type == "restaurant" else 0,  # is restaurant
            1 if place.place_type == "cafe" else 0,  # is cafe
            1 if place.place_type == "bar" else 0,  # is bar
        ]
        place_features.append(features)
    
    # Normalize features
    scaler = StandardScaler()
    place_features = scaler.fit_transform(place_features)
    
    # Calculate user preferences based on their reviews
    user_vector = np.zeros(len(place_features[0]))
    if user_reviews:
        # Calculate average rating given by user
        user_vector[0] = np.mean([r.rating for r in user_reviews])
        # Count number of reviews
        user_vector[1] = len(user_reviews)
        # Count preferences for different place types
        for review in user_reviews:
            place = db.get(Place, review.place_id)
            if place.place_type == "restaurant":
                user_vector[2] += 1
            elif place.place_type == "cafe":
                user_vector[3] += 1
            elif place.place_type == "bar":
                user_vector[4] += 1
    
    # Calculate similarity scores
    similarity_scores = cosine_similarity([user_vector], place_features)[0]
    
    # Get top N recommendations
    top_indices = np.argsort(similarity_scores)[-n_recommendations:][::-1]
    
    # Convert place indices back to place IDs
    place_id_dict = {idx: place.id for idx, place in enumerate(places)}
    recommendations = []
    for idx in top_indices:
        place_id = place_id_dict[idx]
        recommendations.append({
            "user_id": user_id,
            "place_id": place_id,
            "algorithm": RecommendationAlgorithm.TRANSFER_LEARNING.value,
            "score": float(similarity_scores[idx])
        })
    
    return recommendations

def generate_recommendations(
    db: Session,
    user_id: int,
    algorithm: str = "autoencoder",
    n_recommendations: int = 5
) -> List[Dict]:
    """Generate recommendations for a user using the specified algorithm."""
    if algorithm == "autoencoder":
        return autoencoder_recommendations(db, user_id, n_recommendations)
    elif algorithm == "svd":
        return svd_recommendations(db, user_id, n_recommendations)
    elif algorithm == "transfer_learning":
        return transfer_learning_recommendations(db, user_id, n_recommendations)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}") 