"""Recommand some questions to a new student given the tables of marks from create_marks function."""

__author__ = """Kayané Elmayan Robach"""
__email__ = 'kaya.robach@gmail.com'
__version__ = '0.1.0'

from .dropout import dropout_prediction_training_data
from .dropout import dropout_prediction
from .dropout import dropout_recommendation

from .final_recommendation import recom_algorithm

from .recommender import algorithm
from .recommender import get_files