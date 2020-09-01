"""Top-level package for UQDKR LrnXPAnaToolbox."""

__author__ = """Kayané Elmayan Robach"""
__email__ = 'kaya.robach@gmail.com'
__version__ = '0.3.3'

from .lib import str_question_tolist
from .lib import list_question_tostr
from .lib import from_list_to_str
from .lib import from_str_to_list

from .LrnXPAnaToolbox import test

from .cluster_students.cluster_students import plot_clustering
from .cluster_students.cluster_students import creationDFClustering
from .cluster_students.cluster_students import optimal_n_clusters
from .cluster_students.cluster_students import similar_students

from .create_marks.create_marks import conditionnal_proba
from .create_marks.create_marks import complete_proba_matrix
from .create_marks.create_marks import create_marks

from .transform_data.transform_data import to_workable_json
from .transform_data.transform_data import json_to_csv_seq
from .transform_data.transform_data import csv_to_dataframe
from .transform_data.transform_data import dict_diagnostique
from .transform_data.transform_data import json_to_csv_track
from .transform_data.transform_data import remplir_etape
from .transform_data.transform_data import index_essay
from .transform_data.transform_data import id_mpae_3_step
from .transform_data.transform_data import set_essai
from .transform_data.transform_data import duree
from .transform_data.transform_data import plus_proche
from .transform_data.transform_data import get_index_connection
from .transform_data.transform_data import get_parcours_diagnostique
from .transform_data.transform_data import tranche_horaire
from .transform_data.transform_data import transform_data

from .recommender.dropout import dropout_prediction_training_data
from .recommender.dropout import dropout_prediction
from .recommender.dropout import dropout_recommendation

from .recommender.final_recommendation import recom_algorithm

from .recommender.recommender import algorithm
from .recommender.recommender import get_files
