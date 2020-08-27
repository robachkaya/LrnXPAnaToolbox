## EVIDENCEB, 2020
## recommander.py
## File description:
## recommander function

from final_test_recommendation import recom_algorithm
from dropout import final_recommendation_dataset
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
            print("    student_df           File containing data of the student you")
            print("                         want to recommend questions")
            print("    student_marks        File containing marks of the student for recommendations")
            print("    all_students_df      File containing data of all students")
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
    recom_quests = recom_algorithm(student_marks, marks_df)
    quest_after_dropout = final_recommendation_dataset(student_data, students_df, recom_quests)
    if quest_after_dropout.empty:
        quest_after_dropout = recom_quests
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
    #loop(df)

if __name__ == '__main__':
    main(sys.argv)
