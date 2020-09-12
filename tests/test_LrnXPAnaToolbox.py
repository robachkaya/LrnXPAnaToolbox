#!/usr/bin/env python

"""Tests for `LrnXPAnaToolbox` package."""

import unittest
import sklearn
from sklearn import preprocessing
import pandas as pds
import os
import datetime
import LrnXPAnaToolbox as lrn
from LrnXPAnaToolbox import *

class Test_cluster_students(unittest.TestCase):

    def test_creation_df(self):
        count = 0
        csv = pds.read_csv((os.path.join(".", "data", "stud_df_100.csv")))
        data = lrn.cluster_students.cluster_students.creationDFClustering(csv)
        if {'id_eleve','temps_moyen_reponse','taux_reussite','parcours',
        'duree_totale','nbr_questions','jour','horaire'}.issubset(data.columns):
            count += 1
        self.assertEqual(1, count)

    def test_optimal_nbr_cluster(self):
        csv = pds.read_csv((os.path.join(".", "data", "stud_df_100.csv")))
        data = lrn.creationDFClustering(csv)
        length = len(data['id_eleve'].unique())
        DFVariables = data.drop(['id_eleve'], axis = 1)
        DFVariables = DFVariables.apply(pds.to_numeric)
        normalized_data = preprocessing.normalize(DFVariables)
        optimal_clust = lrn.optimal_n_clusters(normalized_data, sklearn.cluster.MiniBatchKMeans, length)
        optimal_clust = optimal_clust[0]
        self.assertGreaterEqual(10, optimal_clust)

    def test_similar_students(self):
        count = 0
        df = pds.read_csv((os.path.join(".", "data", "stud_df_100.csv")))
        new_stud = pds.read_csv((os.path.join(".", "data", "new_stud_test.csv")))
        new_df = lrn.similar_students(df, new_stud, False)
        new_df = pds.DataFrame(new_df[0])
        today = datetime.date.today()
        os.remove((os.path.join(".","data", f"DF_MBKMeans_{today}.pk1")))
        if {'id_eleve', 'Cluster'}.issubset(new_df.columns):
            count += 1
        self.assertEqual(count, 1)

class Test_recommender(unittest.TestCase):

    def test_dropout(self):
        count = 0
        df = pds.read_csv((os.path.join(".", "data", "stud_df_100.csv")))
        new_stud = pds.read_csv((os.path.join(".", "data", "new_stud_test.csv")))
        recom = pds.read_csv((os.path.join(".", "data", "recom.csv")))
        recom['question_id'] = recom['question_id'].astype(str)
        after_dropout = lrn.dropout_recommendation(new_stud, df, recom)
        after_dropout = pds.DataFrame(after_dropout)
        if {'weighted_average_recom_score', 'question_id'}.issubset(after_dropout.columns):
            count += 1
        self.assertEqual(count, 1)

    def test_recom_algorithm(self):
        marks_df = pds.read_csv((os.path.join(".", "data", "stud_marks_100.csv")))
        new_stud_marks = pds.read_csv((os.path.join(".", "data", "new_stud_marks_test.csv")))
        recom_df = lrn.recom_algorithm(new_stud_marks, marks_df)
        self.assertEqual(20, len(recom_df))

class Test_marks_creator(unittest.TestCase):

    def test_marks_table(self):
        count = 0
        df = pds.read_csv((os.path.join(".", "data", "stud_df_100.csv")))
        marks = lrn.marks_table(df)
        if {'student', 'question', 'mark'}.issubset(marks.columns):
            count += 1
        self.assertEqual(1, count)