#!/usr/bin/env python

"""Tests for `LrnXPAnaToolbox` package."""

import unittest
import sklearn
from sklearn import preprocessing
import pandas as pds
import os
import LrnXPAnaToolbox as lrn
from LrnXPAnaToolbox import *

class Test_cluster_students(unittest.TestCase):

    def test_creation_df(self):
        count = 0
        pickle = pds.read_pickle((os.path.join("..", "..", "data", "chatbot_dataset_2020-08-26.pk1")))
        data = lrn.cluster_students.cluster_students.creationDFClustering(pickle)
        if {'id_eleve','temps_moyen_reponse','taux_reussite','parcours',
        'duree_totale','nbr_questions','jour','horaire'}.issubset(data.columns):
            count += 1
        self.assertEqual(1, count)

    def test_optimal_nbr_cluster(self):
        pickle = pds.read_pickle((os.path.join("..", "..", "data", "chatbot_dataset_2020-08-26.pk1")))
        data = lrn.creationDFClustering(pickle)
        length = len(data['id_eleve'].unique())
        DFVariables = data.drop(['id_eleve'], axis = 1)
        DFVariables = DFVariables.apply(pds.to_numeric)
        normalized_data = preprocessing.normalize(DFVariables)
        optimal_clust = lrn.optimal_n_clusters(normalized_data, sklearn.cluster.MiniBatchKMeans, length)
        optimal_clust = optimal_clust[0]
        self.assertGreaterEqual(10, optimal_clust)

    def test_similar_students(self):
        count = 0
        df = pds.read_pickle((os.path.join("..", "..", "data", "chatbot_dataset_2020-08-26.pk1")))
        new_stud = pds.read_pickle((os.path.join("..", "..", "data", "new_student_dataset_2020-08-26.pk1")))
        new_df = lrn.similar_students(df, new_stud, False)
        new_df = pds.DataFrame(new_df[0])
        if {'id_eleve', 'Cluster'}.issubset(new_df.columns):
            count += 1
        self.assertEqual(count, 1)