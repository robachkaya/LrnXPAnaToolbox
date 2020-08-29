## EVIDENCEB, 2020
## dropout.py
## File description:
## dropout function

import sys
from LrnXPAnaToolbox.lib import *
import pandas as pds
import numpy as np
import os
import datetime
from tqdm import tqdm
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

def usage(argv): # argv = arguments passed to the script = dropout.py -h
    if len(argv) == 2:
        if argv[1] == "-h":
            print("\nUSAGE")
            print("    ./dropout.py  new_student_data  available_database  recommendation_dataset")
            print("\nDESCRIPTION")
            print("""    new_student_data       Dataframe with all the data for only one (new) student (created with transform_data.py for example)""")
            print("""    available_database       Dataframe with all the available data (created with transform_data.py for example)""")
            print("""    recommendation_dataset       Dataframe with the N best questions to propose to the new_student""")
            return 1
    return 0

def check_error(argv): # argv = arguments passed to the script = dropout.py new_student_data available_database recommendation_dataset
    if len(argv) != 4:
        print("Wrong number of arguments, try \"-h\" for more informations")
        return 1
    if (argv[1].isidentifier() != True)|(argv[2].isidentifier() != True)|(argv[3].isidentifier() != True):
        print("new_student_data, available_database and recommendation_dataset arguments have to be the name of pandas dataframe so : alphanumeric letters (a-z) and/or (0-9) and/or underscores (_).")
        return 1
    return 0

def dropout_prediction_training_data(dataset, module_concerned, path_concerned, dropout_after_activity):
    """Compute students features to train the dropout prediciton on."""
    module_concerned = int(module_concerned)
    path_concerned = int(path_concerned)
    dropout_after_activity = int(dropout_after_activity)
    d = {}
    grouped = dataset[(dataset.module==module_concerned)&(dataset.path==path_concerned)]
    filtered = grouped.groupby('id_eleve')
    for student, student_groupby in filtered :
        if student_groupby[student_groupby.activity == dropout_after_activity].shape[0] != 0 :
            # the number of the first connection in which the student has done the activity dropout_after_activity
            first_connection_for_dropout_after_activity = student_groupby[student_groupby.activity == dropout_after_activity].sort_values('tps_posix', ascending=True).iloc[0]['num_connection']
            # the date of the last exercise the student has done in the activity dropout_after_activity
            highest_date_dropout_after_activity = max(student_groupby[ student_groupby.activity == dropout_after_activity ]['tps_posix'])
            student_groupby = student_groupby[ (student_groupby.num_connection == first_connection_for_dropout_after_activity) & (student_groupby.tps_posix <= highest_date_dropout_after_activity) ]
            if student_groupby.shape[0] != 0 :
                d[student] = [student]
                # 1st feature : average_response_time
                average_response_time = sum(student_groupby['duree']) / student_groupby.shape[0]
                d[student].append( average_response_time )
                # 2nd feature : success_rate
                success_rate = student_groupby[student_groupby.correct == True].shape[0] / student_groupby.shape[0]
                d[student].append( success_rate )
                # 3rd feature : first assigned_path after the diagnostic test
                if student_groupby[ student_groupby.etape == 2 ].shape[0] == 0:
                    path = 0
                else :
                    path = student_groupby[ student_groupby.etape == 2 ].sort_values('tps_posix',ascending=True).iloc[0]['path']
                d[student].append( path )
                # 4th feature : the day of the fisrt connection in which the student has done the activity
                day = student_groupby.iloc[0]['jour']
                d[student].append( day )
                # 5th feature : the timeslot of the connection
                timeslot = student_groupby.iloc[0]['horaire']
                d[student].append( timeslot )
                # last feature :
                # 0 if the student dropout
                # 1 if the student will reach the next activity : the activity after the activity dropout_after_activity
                if student_groupby[ student_groupby.activity == dropout_after_activity + 1 ].shape[0] == 0 :
                    d[student].append( 0 )
                else :
                    d[student].append( 1 )
    X = np.array(list(d.values()))
    DF = pds.DataFrame(X)
    DF.columns = ['student','average_response_time', 'success_rate', 'assigned_path', 'day', 'timeslot', 'dropout']
    return DF

def dropout_prediction(prediction_test, student_array_features):
    """Return the dropout prediciton value (0 if dropout, 1 if not) and the confidence level of the prediciton.
       The non linear classification method used is the Decision Tree Classifier from sklearn. """
    X = prediction_test.drop('dropout', axis = 1)
    Y = prediction_test['dropout']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
    parametre = [{"max_depth":list(range(2,10))}]
    retry = True
    while retry = True :
        try :
            try_model = GridSearchCV(DecisionTreeClassifier(),parametre,n_jobs=-1)
            try_model.fit(X_train,Y_train)
            retry = False
        except ValueError :
            retry = True
    best_param = try_model.fit(X_train,Y_train).best_params_
    model = DecisionTreeClassifier(max_depth = best_param["max_depth"])
    model.fit(X_train, Y_train)
    return model.predict( student_array_features ), model.score( X_test, Y_test )

def dropout_recommendation(new_student_data, available_database, recommendation_dataset):
    """ parameters :
        - new_student_data  : student for whom we produced 20 best questions
          (there must be the columns : student, duree, correct, etape, jour, horaire)
        - available_database of all the available data : example : chatbot_data.pk1,
          there must be all the data collected but we suppose that the new data of our student are not yet in.
          If there is a way to put them in directly : no need for the parameters new_student_data because we can compute
          this parameters thanks to the parameter : available_database
        - recommendation_dataset of the 20 best questions to propose, columns  ['student_id','question_id','rating'].
        returns :
        - the recommendation_dataset modified according to the predicted dropout prediciton value."""
    # if the available_database contains the student data :
    # we can get the following data for student_array_features with the available_database
    # else : we get student information thanks to the new_student_data parameter
    student = new_student_data['id_eleve']
    student_array_features = np.array([])
    # 1st feature : average_response_time
    average_response_time = sum(new_student_data['duree']) / new_student_data.shape[0]
    student_array_features = np.append( student_array_features, average_response_time )
    # 2nd feature : success_rate
    success_rate = new_student_data[new_student_data.correct == True].shape[0] / new_student_data.shape[0]
    student_array_features = np.append( student_array_features, success_rate )
    # 3rd feature : first assigned_path after the diagnostic test
    if new_student_data[ new_student_data.etape == 2 ].shape[0] == 0:
        path = 0
    else :
        path = new_student_data[ new_student_data.etape == 2 ].sort_values('tps_posix',ascending=True).iloc[0]['path']
    student_array_features = np.append( student_array_features, path )
    # 4th feature : the day of the fisrt connection in which the student has done the activity
    day = new_student_data.iloc[0]['jour']
    student_array_features = np.append( student_array_features, day )
    # 5th feature : the timeslot of the connection
    timeslot = new_student_data.iloc[0]['horaire']
    student_array_features = np.append( student_array_features, timeslot )
    dropout_index = np.array([])
    student_array_features = np.array([student_array_features])
    # question in recommendation_dataset are supposed to be list of the form [module, parcours, activite, exo]

    # to avoid pandas error from manipulating lists in dataframe we convert question into string
    from_list_to_str(recommendation_dataset,'question_id')

    print(f"WARNING : if UserWarning from sklearn\model_selection\_split : not enough participants for the question.")
    for question_id in tqdm(recommendation_dataset['question_id']) :
        prediction_test = dropout_prediction_training_data(available_database, module_concerned=question_id[0] , path_concerned=question_id[1], dropout_after_activity=question_id[2]).drop(['student'], axis = 1)
        # student_dropout = 0 : the student will probably dropout
        # student_dropout = 1 : the student will probably continue
        student_dropout, confidence_rating = dropout_prediction(prediction_test, student_array_features)
        if confidence_rating > 0.86 :
            if student_dropout[0] == str(0) :
                dropout_index = np.append( dropout_index, int(recommendation_dataset[recommendation_dataset['question_id']==question_id].index[0]) )

    from_str_to_list(recommendation_dataset,'question_id')
    new_recommendation_dataset = recommendation_dataset.drop(dropout_index, axis=0) # WARNING : new_recommendation_dataset could be an empty dataframe
    return new_recommendation_dataset

def main(argv):
    if usage(argv) != 0:
        exit(0)
    if check_error(argv) != 0:
        exit(84)
    new_student_data = argv[1]
    available_database = argv[2]
    recommendation_dataset = argv[3]
    dropout_recommendation(new_student_data, available_database, recommendation_dataset)
    return 0

if __name__ == '__main__':
    main(sys.argv)
