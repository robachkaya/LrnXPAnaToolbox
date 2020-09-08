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
        today = datetime.date.today()
        os.remove((os.path.join(".","data", f"DF_MBKMeans_{today}.pk1")))
        if {'id_eleve', 'Cluster'}.issubset(new_df.columns):
            count += 1
        self.assertEqual(count, 1)

class Test_recommender(unittest.TestCase):

    def test_dropout(self):
        count = 0
        df = pds.read_pickle((os.path.join("..", "..", "data", "chatbot_dataset_2020-08-26.pk1")))
        new_stud = pds.read_pickle((os.path.join("..", "..", "data", "new_student_dataset_2020-08-26.pk1")))
        recom = pds.read_csv((os.path.join(".", "data", "recom.csv")))
        recom['question_id'] = recom['question_id'].astype(str)
        after_dropout = lrn.dropout_recommendation(new_stud, df, recom)
        after_dropout = pds.DataFrame(after_dropout)
        if {'weighted_average_recom_score', 'question_id'}.issubset(after_dropout.columns):
            count += 1
        self.assertEqual(count, 1)

    def test_recom_algorithm(self):
        marks_df = pds.read_pickle((os.path.join("..", "..", "data", "table_marks_2020-08-26.pk1")))
        new_stud_marks = pds.read_pickle((os.path.join("..", "..", "data", "new_student_marks_2020-08-26.pk1")))
        marks_df.columns = ['student_id', 'question_id', 'rating']
        new_stud_marks.columns = ['student_id', 'question_id', 'rating']
        recom_df = lrn.recom_algorithm(new_stud_marks, marks_df)
        self.assertEqual(20, len(recom_df))

    def test_recom_after_dropout(self):
        df = pds.read_pickle((os.path.join("..", "..", "data", "chatbot_dataset_2020-08-26.pk1")))
        new_stud = pds.read_pickle((os.path.join("..", "..", "data", "new_student_dataset_2020-08-26.pk1")))
        marks_df = pds.read_pickle((os.path.join("..", "..", "data", "table_marks_2020-08-26.pk1")))
        new_stud_marks = pds.read_pickle((os.path.join("..", "..", "data", "new_student_marks_2020-08-26.pk1")))
        marks_df.columns = ['student_id', 'question_id', 'rating']
        new_stud_marks.columns = ['student_id', 'question_id', 'rating']
        quest_after_dropout = lrn.algorithm(new_stud, new_stud_marks, df, marks_df)
        self.assertGreaterEqual(len(quest_after_dropout), 1)