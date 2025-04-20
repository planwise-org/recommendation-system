"""
Base recommender class that defines the interface for all recommender models.
"""

class BaseRecommender:
    """Base class for all recommender models."""
    
    def get_recommendations(self, user_lat, user_lon, user_prefs, num_recs):
        """
        Get recommendations based on user location and preferences.
        
        Args:
            user_lat (float): User's latitude
            user_lon (float): User's longitude
            user_prefs (dict or array): User preferences
            num_recs (int): Number of recommendations to return
            
        Returns:
            list: Recommendations
        """
        raise NotImplementedError("Subclasses must implement this method.") 