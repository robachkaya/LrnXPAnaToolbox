#!/usr/bin/python3

from recom_test_final import recom_algorithm
from dropout import dropout_recommendation
from cluster_students import similar_students
import pandas as pds
import numpy as np
import os
import sys

def usage(argv):
    if len(argv) == 2:
        if argv[1] == "-h":
            print("USAGE")
            print("    python recommender.py student_df student_marks all_students_df all_student_marks")
            print("DESCRIPTION")
            print("    student_df           File containing datas of the student you")
            print("                         want to recommend questions")
            print("    student_marks        File containing marks of the student for recommendations")
            print("    all_students_df      File containing datas of all students")
            print("    all_students_marks   File containing marks of all students")
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
    print(quest_after_dropout)
    return quest_after_dropout

def get_files(argv):
    df1 = pds.read_pickle((os.path.join(".","data",argv[1])))
    df2 = pds.read_pickle((os.path.join(".","data",argv[2])))
    df3 = pds.read_pickle((os.path.join(".","data",argv[3])))
    df4 = pds.read_pickle((os.path.join(".","data",argv[4])))
    df2.columns = ['student_id', 'question_id', 'rating']
    df4.columns = ['student_id', 'question_id', 'rating']
    return df1, df2, df3, df4

def main(argv):
    if usage(argv) != 0:
        return 0
    if check_error(argv) != 0:
        return 84
    student_data, student_marks, students_df, marks_df = get_files(argv)
    recommendation = algorithm(student_data, student_marks, students_df, marks_df)
    pds.set_option('display.max_columns', None)
    print(f"\n\n\nRECOMMENDATION QUESTION(S) FOR {student_data.iloc[0].id_eleve} : \n\n{recommendation}")
    return 0

if __name__ == '__main__':
    main(sys.argv)


    '''students.columns = ['student_id', 'question_id', 'rating'] # add column 'clusters'
    list_stud = students.student_id.unique()
    for i in range(len(list_stud)) :
        studinput_df = students.loc[students['student_id'] == list_stud[i]]
        students_df = students[students.student_id != list_stud[i]]
    ###################################################
        #recom_algorithm(studinput_df, students_df)
        '''