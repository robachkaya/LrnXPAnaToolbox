#!/usr/bin/python3

from LrnXPAnaToolbox.recommender.final_recommendation import recom_algorithm
from LrnXPAnaToolbox.recommender.dropout import dropout_recommendation
from LrnXPAnaToolbox.cluster_students.cluster_students import similar_students
import pandas as pds
import numpy as np
import os
import sys

def usage(argv):
    if len(argv) == 2:
        if argv[1] == "-h":
            print("\nUSAGE")
            print("    .\recommender.py  student_df  student_marks  all_students_df  all_student_marks")
            print("\nDESCRIPTION")
            print("    student_df           dataframe containing the dataframe of data of the student you")
            print("                         want to recommend questions")
            print("    student_marks        dataframe containing the dataframe of marks of the student for recommendations")
            print("    all_students_df      dataframe containing the dataframe of data of all students")
            print("    all_students_marks   dataframe containing the dataframe of marks of all students")
            print("\nAll files must be in a directory named \'data\'")
            return 1
    return 0

def check_error(argv):
    if len(argv) != 5:
        print("There must be 4 files in argument. Try -h for more information")
        return 1
    if os.path.isdir('data') != True:
        print("File must be in \"data\" directory. Try -h for more informations.")
        return 1
    for i in range(1, len(argv)):
        file = argv[i]
        path = os.path.join(".", "data", file)
        if os.path.isfile(path) != True:
            print("No such file or directory :", file)
            return 1
    return 0

def algorithm(student_data, student_marks, students_df, marks_df):
    """Compute the recommendation algorithm, check the dropout prediciton using only student from the same cluster 
       in order to optimize execution, select the questions for which dropout is not predicted (unless dropout is predicted for all question,
       in this case we just present all the questions without selection) and recommend these questions."""
    cluster_df, student_cluster = similar_students(students_df, student_data, False)
    cluster_df.columns = ['student_id', 'cluster_id']
    student_cluster = int(student_cluster[0])
    new_clust_df = cluster_df[cluster_df.cluster_id == student_cluster]
    students_df = students_df[students_df['id_eleve'].isin(new_clust_df['student_id'])]
    marks_df = marks_df[marks_df['student_id'].isin(new_clust_df['student_id'])]
    recom_quests = recom_algorithm(student_marks, marks_df)
    quest_after_dropout = dropout_recommendation(student_data, students_df, recom_quests)
    if quest_after_dropout.empty:
        quest_after_dropout = recom_quests
    print(f"\n\n\nRECOMMENDATION QUESTION(S) FOR {student_data.iloc[0].id_eleve} : \n\n{quest_after_dropout}")
    return quest_after_dropout

def get_files(argv):
    df1 = pds.read_pickle((os.path.join(".","data",argv[1])))
    df2 = pds.read_pickle((os.path.join(".","data",argv[2])))
    df3 = pds.read_pickle((os.path.join(".","data",argv[3])))
    df4 = pds.read_pickle((os.path.join(".","data",argv[4])))
    df2.columns = ['student_id', 'question_id', 'rating']
    df4.columns = ['student_id', 'question_id', 'rating']
    return df1, df2, df3, df4

student_data = pds.read_pickle((os.path.join(".","data","student_data_2020-08_29.pk1")))
student_marks = pds.read_pickle((os.path.join(".","data","notes_eleve_question.pk1")))
students_df = pds.read_pickle((os.path.join(".","data","chatbot_data_2020-08-29.pk1")))
marks_df = pds.read_pickle((os.path.join(".","data","table_marks_2020-08-26.pk1")))
algorithm(student_data, student_marks, students_df, marks_df)

def main(argv):
    if usage(argv) != 0:
        return 0
    if check_error(argv) != 0:
        return 84
    student_data, student_marks, students_df, marks_df = get_files(argv)
    recommendation = algorithm(student_data, student_marks, students_df, marks_df)
    pds.set_option('display.max_columns', None)
    return 0

if __name__ == '__main__':
    main(sys.argv)