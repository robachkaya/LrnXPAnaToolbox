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

    def test_optimal_nbr_cluster(self):
        pickle = pds.read_pickle((os.path.join("..", "..", "data", "chatbot_dataset_2020-08-26.pk1")))
        data = lrn.cluster_students.cluster_students.creationDFClustering(pickle)
        length = len(data['id_eleve'].unique())
        DFVariables = data.drop(['id_eleve'], axis = 1)
        DFVariables = DFVariables.apply(pds.to_numeric)
        normalized_data = preprocessing.normalize(DFVariables)
        optimal_clust = lrn.cluster_students.cluster_students.optimal_n_clusters(normalized_data, sklearn.cluster.MiniBatchKMeans, length)
        optimal_clust = optimal_clust[0]
        self.assertGreaterEqual(5, optimal_clust)

    def test_

