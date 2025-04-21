"""
Recommenders package for the recommendation system.
Contains implementations of various recommendation algorithms.
"""

from .autoencoder_recommender import AutoencoderRecommender
from .svd_recommender import SVDPlaceRecommender
from .transfer_recommender import TransferRecommender
from .ensemble_recommender import EnsembleRecommender
from .base_recommender import BaseRecommender
from .madrid_transfer_recommender import MadridTransferRecommender

__all__ = [
    'BaseRecommender',
    'AutoencoderRecommender',
    'SVDPlaceRecommender',
    'TransferRecommender',
    'EnsembleRecommender',
    'MadridTransferRecommender'
] 