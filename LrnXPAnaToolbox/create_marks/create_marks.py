## EVIDENCEB, 2020
## create marks for recommendation algorithm
## File description:
## create_marks function

import pandas as pds 
import numpy as np
import os
from tqdm.auto import tqdm 
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

def usage(argv): # argv = arguments passed to the script = create_marks.py -h
    if len(argv) == 2:
        if argv[1] == "-h":
            print("USAGE")
            print("    ./create_marks.py  pickle_file")
            print("DESCRIPTION")
            print("""    pickle_file       File at the pickle format, if the file is called data.pk1, please enter data as pickle_file.""")
            print("""\ncreate_marks.py must be saved in a folder which must contain a data folder with the pickle file :\ndata.pk1""")
            return 1
    return 0

def check_error(argv): # argv = arguments passed to the script = create_marks.py pickle_file
    if len(argv) != 2:
        print("Wrong number of arguments, try \"-h\" for more informations")
        return 1
    if (argv[1].isidentifier() != True) :
        print("Argument have to be the name of a pickle file so : alphanumeric letters (a-z) and/or (0-9) and/or underscores (_).")
        return 1
    return 0

def conditionnal_proba(dataset, module, path, test, failure=False, get_proba_specific_path=False):
    if get_proba_specific_path == True :
        data = dataset[(dataset['module']==module)&(dataset['path']==path)&(dataset['essai']==test)&(dataset['etape']!=3)].loc[:,['id_eleve','module','path','activity','exercice','correct','essai']]
    else :
        data = dataset[(dataset['etape']!=3)&(dataset['essai']==test)][['id_eleve','module','path','activity','exercice','correct','essai']]
    dic_data = {}
    for i in range(data.shape[0]): 
        # for each student as dictionnary key we give a list [module, path, activity, exercise, correct, test number]
        key = data.iloc[i]['id_eleve']
        if key not in dic_data:
            dic_data[key] = []
        dic_data[key].append(tuple(data.iloc[i]))
    P = {}
    for key, values in dic_data.items():
        for i, vi in enumerate(values):
            for j, vj in enumerate(values):
                if i == j:
                    continue
                q1 = vi[1:-1]
                q2 = vj[1:-1]
                couple = q1[:-1], q2[:-1]
                if couple not in P:
                    P[couple] = np.array([ np.zeros((2, 2)) , 0 ])
                # to complete the contingency table :
                # if q1 correct : q1_ok = 1 else q1 uncorrect : q1_ok = 0 :
                q1_ok = 1 if q1[-1] else 0 
                # if q2 correct : q2_ok = 1 else q2 uncorrect : q2_ok = 0 :
                q2_ok = 1 if q2[-1] else 0 
                # [q1_ok,q2_ok] is the box in the grid where we count the number of participants
                P[couple][0][q1_ok, q2_ok] += 1
    for couple in P :
        # reminder : to calcuate the conditionnal probability of column given row we do :
        # probability of column & row / probability of row
        # to compute the probability value :
        if failure == True:
            proba_ij = P[couple][0][0,0]
            proba_i = P[couple][0][0,0] + P[couple][0][0,1]
        else:
            proba_ij = P[couple][0][1,1]
            proba_i = P[couple][0][1,0] + P[couple][0][1,1]
        # reminder : P(answer correctly to q2 given q1 correct ) = P(q1 is correct & q2 is correct) / P(q1 is correct)
        if proba_i != 0 :
            P[couple][1] += proba_ij/proba_i
    return P

def complete_proba_matrix(proba, failure=False):
    couple_list_0 = np.sort(pds.Series([couple[0] for couple in list(proba.keys())]).unique())
    couple_list_1 = np.sort(pds.Series([couple[1] for couple in list(proba.keys())]).unique())
    matrix=[]
    participants = []
    for i,couple in enumerate(proba):
        v = proba[couple]
        matrix.append(dict(row=couple[0],col=couple[1],value=v[1]))
        if failure == True:
            participants.append(dict(row=couple[0],col=couple[1],value=v[0][0,0] + v[0][0,1]))
        else:
            participants.append(dict(row=couple[0],col=couple[1],value=v[0][1,0] + v[0][1,1]))
    return pds.DataFrame(matrix), pds.DataFrame(participants)

def string_to_tuple(string):
    splited = list(map(int,string.split('_')))
    return (splited[0],splited[1],splited[2],splited[3])

def create_marks_1(pickle_file) :
    
    print('preparation step 1 over 4 : computing database')
    database = pds.read_pickle(os.path.join(".","data", pickle_file))
    print('preparation step 2 over 4 : computing conditional probability of success matrix')
    dic_success = conditionnal_proba(database, module=-1, path=-1, test=1, failure=False, get_proba_specific_path=False)
    success_matrix, participants_success = complete_proba_matrix(proba=dic_success, failure=False)
    success_matrix[['row','col']].astype(str)
    data_success = success_matrix.pivot_table(index='row', columns='col', values='value')
    data_success.apply(pds.to_numeric)
    data_success = data_success.fillna(1)
    print('preparation step 3 over 4 : computing conditional probability of failure matrix')
    dic_failure = conditionnal_proba(database, module=-1, path=-1, test=1, failure=True, get_proba_specific_path=False)
    failure_matrix, participants_failure = complete_proba_matrix(proba=dic_failure, failure=True)
    failure_matrix[['row','col']].astype(str)
    data_failure = failure_matrix.pivot_table(index='row', columns='col', values='value')
    data_failure.apply(pds.to_numeric)
    data_failure = data_failure.fillna(1)
    print('preparation step 4 over 4 : computing parameters to create marks')
    DF = database[database['etape']!=3]
    if not type( DF.iloc[0]['id_mpae'] ) == tuple :
        DF['id_mpae'] = DF['id_mpae'].apply(string_to_tuple)
    pos_effect = {question:data_success[question].where(data_success[question]>0.8).index[data_success[question].where(data_success[question]>0.8)==1] for question in DF['id_mpae'].unique()}
    neg_effect = {question:data_failure.T[question].where(data_failure.T[question]>0.8).index[data_failure.T[question].where(data_failure.T[question]>0.8)==1] for question in DF['id_mpae'].unique()}
    q2_time = {question:DF[ (DF.id_mpae==question) & (DF.duree<2700) ]['duree'].describe()['50%'] for question in DF['id_mpae'].unique()}
    q3_time = {question:DF[ (DF.id_mpae==question) & (DF.duree<2700) ]['duree'].describe()['75%'] for question in DF['id_mpae'].unique()}
    
    students = np.array([])
    questions = np.array([])
    marks = np.array([])
    
    dataset = DF.iloc[0:50]
    
    print('creating marks : method 1')

    groupby_student_question = dataset.groupby(['id_eleve','id_mpae'])

    def give_mark(data_student_question):
      
        id_eleve,id_mpae = data_student_question[['id_eleve','id_mpae']].iloc[0]

        mark = 1 

        question_first_test_date = min(data_student_question.tps_posix)
        question_first_test_connection = data_student_question[ data_student_question.tps_posix==question_first_test_date ]['num_connection'].values[0]

        # We look at the first and second test on the first connection for this question :
        question_first_test = data_student_question[ (data_student_question.essai == 1) & (data_student_question.num_connection == question_first_test_connection) ]
        question_second_test = data_student_question[ (data_student_question.essai == 2) & (data_student_question.num_connection == question_first_test_connection) ]

        if question_first_test.shape[0] != 0:
            if question_first_test.shape[0] != 1:
                print("question first test must be 1 line but it's not the case : strange")

            # check the probability for the student to dropout :
            student_tried_idmpae = data_student_question[ data_student_question.essai==1 ]
            question_connections = np.array(student_tried_idmpae['num_connection'].unique())
            question_dates = np.array(student_tried_idmpae['tps_posix'].unique())

            #check same size :
            if len(question_connections)!=len(question_dates):
                print('pb')

            # check if the student didnt try any other exercise after the question 
            df_student_other_question = dataset[ (dataset.id_eleve==id_eleve) & (dataset.id_mpae!=id_mpae) ]
            def student_didnt_try_anything_after(i_connection):
                return df_student_other_question[ (df_student_other_question.num_connection==question_connections[i_connection]) & (df_student_other_question.tps_posix>question_dates[i_connection]) ].shape[0] == 0

            student_didnt_try_anything_after_vectorize = np.vectorize(student_didnt_try_anything_after)

            idx_question_connections = np.arange(len(question_connections))

            # we apply student_didnt_try_anything_after for all the index of the array question_connections
            # to get the array of True False where the condition student_didnt_try_anything_after() is satisfied
            student_didnt_try_anything_after = sum(student_didnt_try_anything_after_vectorize(idx_question_connections))

            probability_of_dropout = student_didnt_try_anything_after / student_tried_idmpae.shape[0]
            if probability_of_dropout > 0.5 :
                mark -= 1
            else :
                mark += 1 

            # test if the average reponse time is quite long whereas the student got right at the first try
            if question_first_test['correct'].iloc[0] == True:
                if question_first_test['tps_posix'].iloc[0] > (q3_time[id_mpae]+q2_time[id_mpae]) / 2 :
                    mark += 1

            else:
                # test if the student got right at the second try  
                if question_second_test.shape[0] != 0 :
                    if question_second_test['correct'].iloc[0] == True :
                        mark += 1

            # test if the student failed any question that has a negative influence at a previous date and succeeded here (at 1st or 2nd test)
            if dataset[ (dataset.id_eleve == id_eleve) & (dataset.correct == False) & (dataset.tps_posix <= question_first_test_date) ]['id_mpae'].isin(neg_effect).sum() > 1 :
                if question_first_test['correct'].iloc[0] == True :
                    mark += 1
                elif question_second_test.shape[0] != 0 :
                    if question_second_test['correct'].iloc[0] == True :
                        mark += 1

            # test if the student succeeded any question that has a positive influence at a previous date and failed here (at 1st or 2nd test)
            if dataset[ (dataset.id_eleve == id_eleve) & (dataset.correct == True) & (dataset.tps_posix <= question_first_test_date) ]['id_mpae'].isin(pos_effect).sum() > 1 :
                if question_first_test['correct'].iloc[0] == False :
                    mark += 1
                elif question_second_test.shape[0] != 0 :
                    if question_second_test['correct'].iloc[0] == False :
                        mark += 1 

        return id_eleve, str(tuple(id_mpae)), int(mark)
    
    tqdm.pandas()
    resultat = groupby_student_question.progress_apply(give_mark)
    
    return pds.DataFrame(list(resultat), columns = ['student', 'question', 'mark'])

def create_marks_2(pickle_file):
    
    print('preparation step 1 over 4 : computing database')
    database = pds.read_pickle(os.path.join(".","data", pickle_file))
    print('preparation step 2 over 4 : computing conditional probability of success matrix')
    dic_success = conditionnal_proba(database, module=-1, path=-1, test=1, failure=False, get_proba_specific_path=False)
    success_matrix, participants_success = complete_proba_matrix(proba=dic_success, failure=False)
    success_matrix[['row','col']].astype(str)
    data_success = success_matrix.pivot_table(index='row', columns='col', values='value')
    data_success.apply(pds.to_numeric)
    data_success = data_success.fillna(1)
    print('preparation step 3 over 4 : computing conditional probability of failure matrix')
    dic_failure = conditionnal_proba(database, module=-1, path=-1, test=1, failure=True, get_proba_specific_path=False)
    failure_matrix, participants_failure = complete_proba_matrix(proba=dic_failure, failure=True)
    failure_matrix[['row','col']].astype(str)
    data_failure = failure_matrix.pivot_table(index='row', columns='col', values='value')
    data_failure.apply(pds.to_numeric)
    data_failure = data_failure.fillna(1)
    print('preparation step 4 over 4 : computing parameters to create marks')
    DF = database[database['etape']!=3]
    if not type( DF.iloc[0]['id_mpae'] ) == tuple :
        DF['id_mpae'] = DF['id_mpae'].apply(string_to_tuple)
    pos_effect = {question:data_success[question].where(data_success[question]>0.8).index[data_success[question].where(data_success[question]>0.8)==1] for question in DF['id_mpae'].unique()}
    neg_effect = {question:data_failure.T[question].where(data_failure.T[question]>0.8).index[data_failure.T[question].where(data_failure.T[question]>0.8)==1] for question in DF['id_mpae'].unique()}
    q2_time = {question:DF[ (DF.id_mpae==question) & (DF.duree<2700) ]['duree'].describe()['50%'] for question in DF['id_mpae'].unique()}
    q3_time = {question:DF[ (DF.id_mpae==question) & (DF.duree<2700) ]['duree'].describe()['75%'] for question in DF['id_mpae'].unique()}
    
    students = np.array([])
    questions = np.array([])
    marks = np.array([]) 
    
    dataset = DF.iloc[0:50]
    
    print('creating marks : method 2')
    
    for student in tqdm(dataset['id_eleve'].unique()):
        
        student_data = dataset[dataset['id_eleve']==student] 
        
        for element_groupby in student_data.groupby('id_mpae'):
            
            data_student_question = element_groupby[1]
            id_eleve,id_mpae = data_student_question[['id_eleve','id_mpae']].iloc[0].values
            
            students = np.append(students, id_eleve)
            questions = np.append(questions, str(tuple(id_mpae)))
    
            mark = 1
        
            question_first_test_date = min(data_student_question.tps_posix)
            question_first_test_connection = data_student_question[ data_student_question.tps_posix==question_first_test_date ]['num_connection'].values[0]
            
             # We look at the first and second test on the first connection for this question :
            question_first_test = data_student_question[ (data_student_question.essai == 1) & (data_student_question.num_connection == question_first_test_connection) ]
            question_second_test = data_student_question[ (data_student_question.essai == 2) & (data_student_question.num_connection == question_first_test_connection) ]

            # check the probability for the student to dropout :
            question_connections = np.array(question_first_test['num_connection'].unique())
            question_dates = np.array(question_first_test['tps_posix'].unique())

            #check same size :
            if len(question_connections)!=len(question_dates):
                print('pb')

            # check if the student didnt try any other exercise after the question 
            df_student_other_question = student_data[ (student_data.id_mpae!=id_mpae) ]
            def student_didnt_try_anything_after(i_connection):
                return df_student_other_question[ (df_student_other_question.num_connection==question_connections[i_connection]) & (df_student_other_question.tps_posix>question_dates[i_connection]) ].shape[0] == 0

            student_didnt_try_anything_after_vectorize = np.vectorize(student_didnt_try_anything_after)

            idx_question_connections = np.arange(len(question_connections))

            # we apply student_didnt_try_anything_after for all the index of the array question_connections
            # to get the array of True False where the condition student_didnt_try_anything_after() is satisfied
            student_didnt_try_anything_after = sum(student_didnt_try_anything_after_vectorize(idx_question_connections))

            probability_of_dropout = student_didnt_try_anything_after / question_first_test.shape[0]
            if probability_of_dropout > 0.5 :
                mark -= 1
            else :
                mark += 1
            
            if question_first_test['correct'].any() == True:
                if question_first_test['tps_posix'].iloc[0] > (q3_time[id_mpae]+q2_time[id_mpae]) / 2 :
                    mark += 1
            else :
                if question_second_test['correct'].any()==True:
                    mark += 1
                        
            # test if the student failed any question that has a negative influence at a previous date and succeeded here (at 1st or 2nd test)
            if student_data[ (student_data.correct == False) & (student_data.tps_posix <= question_first_test_date) ]['id_mpae'].isin(neg_effect).sum() > 1 :
                if question_first_test['correct'].iloc[0] == True :
                    mark += 1
                elif question_second_test.shape[0] != 0 :
                    if question_second_test['correct'].iloc[0] == True :
                        mark += 1

            # test if the student succeeded any question that has a positive influence at a previous date and failed here (at 1st or 2nd test)
            if student_data[ (student_data.correct == True) & (student_data.tps_posix <= question_first_test_date) ]['id_mpae'].isin(pos_effect).sum() > 1 :
                if question_first_test['correct'].iloc[0] == False :
                    mark += 1
                elif question_second_test.shape[0] != 0 :
                    if question_second_test['correct'].iloc[0] == False :
                        mark += 1 

            marks = np.append(marks, int(mark))
            
        d = [[e,q,int(n)] for e,q,n in zip(students,questions,marks)]
                          
    return pds.DataFrame(d, columns = ['student', 'question', 'mark'])

def main(argv):
    if usage(argv) != 0:
        exit(0)
    if check_error(argv) != 0:
        exit(84)
    database = argv[1]
    create_marks_2(database)
    return 0

if __name__ == '__main__':
    main(sys.argv)