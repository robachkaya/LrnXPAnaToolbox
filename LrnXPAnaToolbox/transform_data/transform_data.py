## EVIDENCEB, 2020
## transform_data.py
## File description:
## transform data function

import ijson
import json
import csv
import pandas as pds
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import datetime
import scipy
import itertools
import os
import io
import sys
import time
import datetime
from math import *

def usage(argv): # argv = arguments passed to the script = transform_data.py -h
    if len(argv) == 2:
        if argv[1] == "-h":
            print("\nUSAGE")
            print("    ./transform_data.py  option  original_json_sequences_name  original_json_trackings_name")
            print("\nDESCRIPTION")
            print("""    option = 3       Transform data and save 2 pandas dataframes : sequences and chatbot""")
            print("""    option = 1       Transform data and save 1 pandas dataframe : sequences""")
            print("""    option = 2       Transform data and save 1 pandas dataframe : chatbot""")
            print("""    original_json_sequences_name   original_json_trackings_name   are the names of both json files.""")
            print("""\ntransform_data.py must be into a directory with a directory named \'data\' with the files : \noriginal_json_sequences_name.txt and original_json_sequences_name.json\noriginal_json_trackings_name.txt and original_json_trackings_name.json""")
            return 1
    return 0

def check_error(argv): # argv = arguments passed to the script = transform_data.py option seq track
    if len(argv) != 4:
        print("Wrong number of arguments, try \"-h\" for more informations")
        return 1
    if (argv[1].isdigit() != True) :
        print("Arguments have to be numerical characters (1, 2 or 3).")
        return 1
    if argv[1].isdigit() :
        if not int(argv[1]) in [1,2,3] :
            print("Argument must be 3 if you want to save both : sequences and trackings pandas dataframes.")
            print("Argument must be 1 if you want to save the sequences pandas dataframe.")
            print("Argument must be 2 if you want to save the trackings pandas dataframe.")
            return 1
    if (argv[2].isidentifier() != True)|(argv[3].isidentifier() != True):
        print("names arguments have to be the names of the sequences and trackings json files so : alphanumeric letters (a-z) and/or (0-9) and/or underscores (_).")
        return 1
    return 0

def to_workable_json(namefile):
    """Modify the json file (at the text format) to get a workable json file (at the json format) under the name transformed...json."""
    f = io.open((os.path.join(".","data", "{0}.txt".format(namefile))), 'r', encoding='utf-8')
    content = f.read()
    first_elements, last_elements = content[0:2],content[-3:-1]
    check = (first_elements=="[{") & (last_elements=="}]")
    if check :
        print("The json seems to have the correct format : Let's transform the json")
        content = content[content.find('['):]
        element = "["
        to_insert1 = """{ "data" :"""
        to_insert2 = "}"
        idx = content.index(element)
        content = content[:idx] + to_insert1 + content[idx:] + to_insert2
        result = str(content)
        new_content = io.open((os.path.join(".","data", "transformed_{0}.json".format(namefile))), 'w+', encoding='utf-8')
        new_content.write(result)
        new_content.close()
        print("json transformed")
    else:
        print("The json seems to not have the correct format : Let's try to transform the json")
        content = content[content.find('['):]
        element = "["
        to_insert1 = """{ "data" :"""
        to_insert2 = "}"
        idx = content.index(element)
        content = content[:idx] + to_insert1 + content[idx:] + to_insert2
        result = str(content)
        new_content = io.open((os.path.join(".","data", "transformed_{0}.json".format(namefile))), 'w+', encoding='utf-8')
        new_content.write(result)
        new_content.close()
        print("json transformed")

def json_to_csv_seq(csvfile, jsonfile):
    """Take the json modified from sequences file from the previous function to writw a csv file."""
    with io.open((os.path.join(".","data", "{0}.csv".format(csvfile))), 'w', encoding='utf_8') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        with io.open(os.path.join(".","data", "{0}.json".format(jsonfile)), encoding='utf_8') as f:
            objects = ijson.items(f, '')
            for datum in objects :
                for i in range(len(list(datum.values())[0])):
                    multi_rows = list(datum.values())[0][i]
                    try :
                        module = multi_rows["module"]
                    except KeyError:
                        module = np.nan
                    try:
                        path = multi_rows["path"]
                    except KeyError:
                        path = np.nan
                    try:
                        activity = multi_rows["activity"]
                    except KeyError:
                        activity = np.nan
                    try :
                        token = multi_rows["token"]
                    except KeyError:
                        token = np.nan
                    for sequence in json.loads(multi_rows["sequence"]):
                        try:
                            real_module = sequence['module']
                        except KeyError:
                            real_module = np.nan
                        try:
                            real_path = sequence['path']
                        except KeyError:
                            real_path = np.nan
                        try:
                            real_activity = sequence['activity']
                        except KeyError:
                            real_activity = np.nan
                        try:
                            real_exercice = sequence['exercice']
                        except KeyError:
                            real_exercice = np.nan
                        try:
                            difficultyseq = sequence['difficulty']
                        except KeyError:
                            difficultyseq = np.nan
                        try:
                            type_answer = sequence["type"]
                        except KeyError:
                            type_answer = np.nan
                        try:
                            reponse_eleve = sequence["content_id"]
                        except KeyError:
                            reponse_eleve = np.nan
                        try:
                            reponse_correcte = sequence["correct_answer"]
                        except KeyError:
                            reponse_correcte = np.nan

                        spamwriter.writerow([real_module, real_path, real_activity, real_exercice, difficultyseq, type_answer, module, path, activity, token, reponse_eleve, reponse_correcte])

def csv_to_dataframe(csvfile,columns_names):
    """Transform the csv file from the previous functio into a dataframe."""
    return pds.read_csv((os.path.join(".","data", "{0}.csv".format(csvfile))),names=columns_names)

def dict_diagnostique(dataframe_chatbot):
    """Create a dictionnary with : for each question from the diagnostic test (id_mpae) in key, the path corresponding in value."""
    chatbot_data_difficulty = dataframe_chatbot[dataframe_chatbot['real_path']==0]
    chatbot_data_difficulty = pds.DataFrame.set_index(chatbot_data_difficulty,(np.arange(len(chatbot_data_difficulty))))
    diagnostique = {}

    # we have to transform the id_mpae column in string :
    dfnew = chatbot_data_difficulty[['id_mpae']].copy(deep=True)
    pds.options.mode.chained_assignment = None
    chatbot_data_difficulty['id_mpae'] = dfnew['id_mpae'].apply( lambda x : ''.join(str(e) for e in x) )

    for idmpae in chatbot_data_difficulty['id_mpae'].unique():
        diagnostique[idmpae]=[]
        df = chatbot_data_difficulty[chatbot_data_difficulty['id_mpae']==idmpae]['difficultyseq']
        for val in df:
            try :
                json.loads(val)
                diagnostique[idmpae].append(val)
                break
            except ValueError:
                continue
            except TypeError:
                continue
    cpt=0
    for key in diagnostique:
        if not diagnostique[key]:
            cpt+=1
            print(f"None of the difficulty dictionnary is readable with json.loads for the exercice {key}, you must code a solution to transform the string into a workable dictionnary.")
    if cpt==0:
        for key in diagnostique:
            diagnostique[key]=diagnostique[key][0]
    return diagnostique

def json_to_csv_track(csvfile, jsonfile):
    """Take the json modified from trackings file from the previous function to writw a csv file."""
    with io.open((os.path.join(".","data", "{0}.csv".format(csvfile))), 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        with io.open((os.path.join(".","data", "{0}.json".format(jsonfile)))) as f:
            objects = ijson.items(f, '')
            for datum in objects :
                for i in range(len(list(datum.values())[0])):
                    multi_rows = list(datum.values())[0][i]
                    module = multi_rows["module"]
                    path = multi_rows["path"]
                    activity = multi_rows["activity"]
                    token = multi_rows["token"]
                    for tracking in json.loads(multi_rows["tracking"]):
                        try:
                            tms = tracking["tms"]
                        except KeyError:
                            tms = np.nan
                        try:
                            exerciceId = tracking["exerciceId"]
                        except KeyError:
                            exerciceId = np.nan
                        try:
                            isCorrect = tracking["isCorrect"]
                        except KeyError:
                            isCorrect = np.nan
                        try:
                            question_type = tracking["type"]
                        except KeyError:
                            question_type = np.nan

                        spamwriter.writerow([token, module, path, activity, tms, exerciceId, isCorrect, question_type])
                        

def remplir_etape():
    """L1, L2, L3 are the list of the questions from each of the different steps : 1 for the diagnostic test, 2 for the exercice in the path assigned, 3 for the final test.
       1st step : path and activity = 0
       3rd step : the last activity of each path in each module + 1
       2nd step : other question"""
    # WARNING : must be update for each version of the chatbot
    d={}
    L3=set([(1,1,7),(1,2,6),(1,3,6),(2,1,6),(2,2,5),(2,3,5),(3,1,6),(3,2,4),(3,3,6),(4,1,6),(4,2,4),(4,3,6),(5,1,6),(5,2,6),(5,3,5),(6,1,5),(6,2,5),(6,3,4),(7,1,6),(7,2,7),(7,3,6),(8,1,6),(8,2,4),(8,3,5),(9,1,4),(9,2,5),(9,3,5)])
    for triple in L3:
        d[triple]=3
    L1=set([(1,0,0),(2,0,0),(3,0,0),(4,0,0),(5,0,0),(6,0,0),(7,0,0),(8,0,0),(9,0,0)])
    for triple in L1:
        d[triple]=1
    L2 = set(list(itertools.product(*[[1,2,3,4,5,6,7,8,9],[0,1,2,3],[0,1,2,3,4,5,6,7,]])))
    for triple in L2-L1-L3 :
        d[triple]=2
    return d

def index_essay(df):
    colonne_to_sort = 'tms'
    df = df.sort_values(colonne_to_sort, axis = 0, ascending = True)
    df = pds.DataFrame.set_index(df,(np.arange(len(df))+1))
    df['essai'] = df.index
    return df

def id_mpae_3_step(row):
    if row.etape==3 :
        return [int(-1),int(-1),int(-1),int(-1)]
    else :
        return [int(row.module),int(row.path),int(row.activity),int(row.exerciceId)]

def set_essai(data):
    """For each student and question (id_mpae) we have to number the test done (according to the date).
       For each student and question we isolate a dataframe ; then we sort it by ascending date (tms) and we attribute the index as the test number."""
    # we have to transform the id_mpae column in string :
    dfnew = data[['id_mpae']].copy(deep=True)
    pds.options.mode.chained_assignment = None
    data['id_mpae'] = dfnew['id_mpae'].apply( lambda x : ''.join(str(e) for e in x) )

    def create_frames(data):
        frames = []
        data_eleve_1_2 = data[(data['etape']!=3)].groupby(['token','id_mpae'])
        data_eleve_3 = data[(data['etape']==3)]
        indexed_data_1_2 = data_eleve_1_2.apply(index_essay)
        #indexed_data_3 = data_eleve_3.apply(index_essay)
        frames.append(indexed_data_1_2)
        frames.append(data_eleve_3)
        return frames

    result = pds.concat( create_frames(data) )
    result = pds.DataFrame.set_index(result,np.arange(result.shape[0]))

    result['id_mpae'] = result.apply(lambda row : id_mpae_3_step(row), axis=1)
    return result

def duree(data):
    """For each student, we look at the date of each line of exercice and we calculate the duration according to the next date.
       The last duration cannot be calculated because there is no next date."""
    frames = []
    for eleve in data['token'].unique():
        data_eleve = data[data['token']==eleve]
        data_eleve = data_eleve.sort_values('tms', axis = 0, ascending = True)
        data_eleve = pds.DataFrame.set_index(data_eleve,np.arange(data_eleve.shape[0]))
        data_eleve_withoutfirst = data_eleve.drop([0],axis=0)
        data_eleve_withoutfirst = data_eleve_withoutfirst.append(data_eleve.iloc[0])
        data_eleve_withoutfirst = pds.DataFrame.set_index(data_eleve_withoutfirst,np.arange(data_eleve_withoutfirst.shape[0]))
        data_eleve_withoutfirst = data_eleve_withoutfirst.sort_values('tms', axis = 0, ascending = True)
        data_eleve['tps+1'] = data_eleve_withoutfirst['tms']
        data_eleve['duree'] = data_eleve['tps+1']-data_eleve['tms']
        data_eleve = pds.DataFrame.set_index(data_eleve,np.arange(data_eleve.shape[0]))
        data_eleve = data_eleve.drop([data_eleve.shape[0]-1],axis=0)
        frames.append(data_eleve)
    result = pds.concat(frames)
    result = pds.DataFrame.set_index(result,np.arange(result.shape[0]))
    return result

def plus_proche(idx,array):
    new_array = abs(np.array(array) - idx)
    result = np.argmin(new_array)
    if idx - array[result] > 0:
        return result + 1
    else:
        return result

def get_index_connection(data):
    """To get the index connection, we look at the big duration (indicating that the next exercise has been done later : at a next connection).
       We save the index of these lines in the dataframe and we look at other questions.
       If the question is done before a big duration, the question is done during the connection ending with this big duration.
       If the question is done later, we are in a new connection."""
    data = data.sort_values('tms', axis=0, ascending=True)
    data = pds.DataFrame.set_index(data,np.arange(data.shape[0]))
    eleve_index_grosses_durees = {}
    for idt in data['token'].unique():
        eleve_index_grosses_durees[idt] = data[(data['duree']>=2700)&(data['token']==idt)].index , data[data['token']==idt].index
    index_connection = {idx:1 for idx in range(data.shape[0])}
    for cle in eleve_index_grosses_durees:
        if len(eleve_index_grosses_durees[cle][0])!=0:
            for idx in eleve_index_grosses_durees[cle][1]:
                idx_proche = plus_proche(idx , eleve_index_grosses_durees[cle][0])
                index_connection[idx] = idx_proche + 1
    return index_connection

def get_parcours_diagnostique(dico):
    """We attribute the diagnotic path of any question of the 1st step looking at the dictionnary of difficulty."""
    if type(dico)==str:
        dico = json.loads(dico)
        for key in dico:
            if dico[key]==1:
                key=key.strip()
                return int(key[4])
    else:
        return np.nan

def tranche_horaire(data):
    """For each student and each connection we look at the major hour in each the connection took place."""
    d={}
    data_eleve_connection = data.groupby(['token','num_connection'])
    for serie in data_eleve_connection :
        df = serie[1].sort_values('tms', axis=0, ascending=True)
        duree = np.array(df['duree'])
        array_horaires = np.array(df['tms'])
        for indice in range(len(array_horaires)):
            array_horaires[indice] = int(datetime.datetime.fromtimestamp(array_horaires[indice]).hour)
        d[serie[0]] = np.quantile(array_horaires,0.5)
    return d


def data_transformation(option,original_json_sequences_name,original_json_trackings_name):

    today = datetime.date.today()

    if option == 3: #save 2 pandas dataframe : sequences and trackings

        print("building sequences data\n")
        to_workable_json(original_json_sequences_name)
        json_to_csv_seq('jsonfile_to_csv_sequences2',f'transformed_{original_json_sequences_name}')
        print("                                              ,","                                             ,o","                                             :o",sep="\n")
        dumpseq_1 = csv_to_dataframe('jsonfile_to_csv_sequences2',['real_module','real_path','real_activity','real_exercice','difficultyseq', 'type_answer', 'module','path','activity','id_eleve', 'reponse_eleve', 'reponse_correcte'])
        print("                    _....._                  `:o","                  .'       ``-.                \o","                 /  _      _   \                \o",sep="\n")
        dumpseq_1[['real_module','real_path','real_activity','real_exercice']] = dumpseq_1[['real_module','real_path','real_activity','real_exercice']].apply(pds.to_numeric, errors='coerce')
        print("                :  /*\    /*\  :                 ;o","                |  \_/    \_/  :                  ;o","                (       U      /                  ;o",sep="\n")
        dumpseq_1[['real_module','real_path','real_activity','real_exercice']] = dumpseq_1[['real_module','real_path','real_activity','real_exercice']].astype('Int16')
        print("                 \  (\_____/) /                  /o","                  \   UQDKR  (                  /o","                   \         (                ,o:",sep="\n")
        dumpseq_1 = dumpseq_1.fillna(-1)
        print("                   )          \,           .o;o'           ,o'o'o.","                 ./          /\o;o,,,,,;o;o;''         _,-o,-'''-o:o.","  .             ./o./)        \    'o'o'o''         _,-'o,o'         o",sep="\n")
        dumpseq_1.loc[:,'id_mpae'] = dumpseq_1.apply(lambda row: [int(row.module),int(row.path),int(row.activity),int(row.real_exercice)], axis=1)
        print("  o           ./o./ /       .o \.              __,-o o,o'","  \o.       ,/o /  /o/)     | o o'-..____,,-o'o o_o-'","  `o:o...-o,o-' ,o,/ |     \   'o.o_o_o_o,o--''",sep="\n")
        dumpseq_1.loc[:,'id_final_mpae'] = dumpseq_1.apply(lambda row: [int(row.real_module),int(row.real_path),int(row.real_activity),int(row.real_exercice)], axis=1)
        print("  .,  ``o-o'  ,.oo/   'o /\.o`.","  `o`o-....o'o,-'   /o /   \o \.                       ,o..         o","    ``o-o.o--      /o /      \o.o--..          ,,,o-o'o.--o:o:o,,..:o",sep="\n")
        diagnostique = dict_diagnostique(dumpseq_1)
        print("                  (oo(          `--o.o`o---o'o'o,o,-'''        o'o'o","                   \ o\              ``-o-o''''","    ,-o;o           \o \\","    ,-o;o           \o \\",sep="\n")

        # pandas error with list we have to transform the id_mpae column in string :
        dfnew = dumpseq_1[['id_mpae']].copy(deep=True)
        pds.options.mode.chained_assignment = None
        dumpseq_1['id_mpae'] = dfnew['id_mpae'].apply( lambda x : ''.join(str(e) for e in x) )
        dumpseq_1['diagnostique'] = dumpseq_1['id_mpae'].map(diagnostique)
        dumpseq_1['id_mpae'] = dfnew['id_mpae'].apply( lambda x : list( int(e) for e in x ) )

        print("   /o/               )o )","  (o(               /o /","   \o\.       ...-o'o /",sep="\n")
        dumpseq_1.to_pickle((os.path.join(".","data", f"sequences_data_{today}.pk1")))
        print("    \ o`o`-o'o o,o,--'","       ```o--'''",sep="\n")
        print("\nsequences dataframe built\n")
        print("building trackings data\n")
        to_workable_json(original_json_trackings_name)
        json_to_csv_track('jsonfile_to_csv_trackings2',f'transformed_{original_json_trackings_name}')
        print("                                                       ______","                                                     /    /_-._",sep="\n")
        print("                                                   / /_ ~~o\  :Y","                                                   / : \~x.  ` ')",sep="\n")
        dumptrack_2 = csv_to_dataframe('jsonfile_to_csv_trackings2',['token','module','path','activity','tms','exerciceId','isCorrect','question_type'])
        dico_etapes = remplir_etape()
        print("                                                  /  |  Y< ~-.__j")
        dumptrack_2['tms'] = dumptrack_2['tms']/1000
        dumptrack_2['jour'] = dumptrack_2.apply(lambda row: datetime.datetime.fromtimestamp(row.tms).weekday(), axis=1)
        print("                                                 UQDKR  l<  /.-~")
        dumptrack_2['etape'] = dumptrack_2.apply(lambda row: dico_etapes[(row.module,row.path,row.activity)],axis=1)
        print("                                                 ` l /~\ \<|Y")
        dumptrack_2['id_mpa'] = dumptrack_2.apply(lambda row: [int(row.module),int(row.path),int(row.activity)], axis=1)
        print("                                                    ',-~\ \L|")
        dumptrack_2['id_mpae'] = dumptrack_2.apply(lambda row: id_mpae_3_step(row), axis=1)
        print("'--.____  '--------.______       _.----.-----./      :/   '--'")
        dumptrack_2['date'] = dumptrack_2.apply(lambda row: (datetime.datetime.fromtimestamp(row.tms).year,datetime.datetime.fromtimestamp(row.tms).month,datetime.datetime.fromtimestamp(row.tms).day), axis=1)
        print("        '--.__            `'----/       '-.      __ :/")
        chatbot = set_essai(dumptrack_2)
        print("              '-.___           :           \   .'  )/")
        
        # we have to transform the id_mpae column in string :
        dfnew = chatbot[['id_mpae']].copy(deep=True)
        pds.options.mode.chained_assignment = None
        chatbot['id_mpae'] = dfnew['id_mpae'].apply( lambda x : ''.join(str(e) for e in x) )
        chatbot['diagnostique'] = chatbot['id_mpae'].map(diagnostique)
        chatbot['id_mpae'] = dfnew['id_mpae'].apply( lambda x : list( int(e) for e in x ) )

        chatbot.loc[chatbot['etape']!=1,'diagnostique'] = np.nan
        print("                    '---._           _.-'   ] /  _/")
        chatbot_new = duree(chatbot)
        print("                         '-._      _/     _/ / _/")
        chatbot_new = chatbot_new.sort_values('tms', axis=0, ascending=True)
        chatbot_new = pds.DataFrame.set_index(chatbot_new,np.arange(chatbot_new.shape[0]))
        print("                             \_ .-'____.-'__< |  \___")
        connection_index = get_index_connection(chatbot_new)
        chatbot_new['num_connection'] = chatbot_new.index.map(connection_index)
        print("                               <_______.\    \_\_---.7")
        came_back = {}
        for idt in chatbot_new['token'].unique() :
            came_back[idt] = sum(chatbot_new[chatbot_new['token']==idt]['duree'] > 2700)
        print("                              |   /'=r_.-'     _\\ =/")
        chatbot_new['reconnections'] = chatbot_new['token'].map(came_back)
        print("                          .--'   /            ._/'>")
        nbr_questions = {}
        for idt in chatbot_new['token'].unique() :
            nbr_questions[idt] = chatbot_new[chatbot_new['token']==idt].shape[0]
        print("                        .'   _.-'")
        chatbot_new['parcours_diagnostique'] = chatbot_new['diagnostique'].apply(get_parcours_diagnostique)
        chatbot_new['nbr_questions_faites'] = chatbot_new['token'].map(nbr_questions)
        print("                       / .--'")
        horaire = tranche_horaire(chatbot_new)
        print("                      /,/")
        chatbot_new['horaire'] = chatbot_new.apply( lambda row: horaire[(row.token,row.num_connection)],axis=1 )
        chatbot_new.columns = ['id_eleve', 'module', 'path', 'activity', 'tps_posix', 'exercice', 'correct', 'question_type',
       'jour', 'etape', 'id_mpa', 'id_mpae', 'date', 'essai', 'diagnostique',
       'tps+1', 'duree', 'num_connection', 'reconnections',
       'parcours_diagnostique', 'nbr_questions_faites', 'horaire']
        print("                      |/`)")
        chatbot_final = chatbot_new.loc[:,['id_eleve', 'module', 'path', 'activity', 'tps_posix', 'exercice','correct', 'question_type',
       'jour', 'etape', 'id_mpa', 'id_mpae', 'date', 'essai',
       'tps+1', 'duree', 'num_connection', 'reconnections',
       'parcours_diagnostique', 'nbr_questions_faites', 'horaire']]
        print("                      'c=,")
        chatbot_final.to_pickle((os.path.join(".","data", f"chatbot_data_{today}.pk1")))
        print("\nchatbot dataframe built from trackings and sequences data")

    if option == 1: # save 1 pandas dataframe : sequences

        print("building sequences dataframe\n")
        to_workable_json(original_json_sequences_name)
        json_to_csv_seq('jsonfile_to_csv_sequences2',f'transformed_{original_json_sequences_name}')
        print("                                              ,","                                             ,o","                                             :o",sep="\n")
        dumpseq_1 = csv_to_dataframe('jsonfile_to_csv_sequences2',['real_module','real_path','real_activity','real_exercice','difficultyseq', 'type_answer','module','path','activity','id_eleve', 'reponse_eleve', 'reponse_correcte'])
        print("                    _....._                  `:o","                  .'       ``-.                \o","                 /  _      _   \                \o",sep="\n")
        dumpseq_1[['real_module','real_path','real_activity','real_exercice']] = dumpseq_1[['real_module','real_path','real_activity','real_exercice']].apply(pds.to_numeric, errors='coerce')
        print("                :  /*\    /*\  :                 ;o","                |  \_/    \_/  :                  ;o","                (       U      /                  ;o",sep="\n")
        dumpseq_1[['real_module','real_path','real_activity','real_exercice']] = dumpseq_1[['real_module','real_path','real_activity','real_exercice']].astype('Int16')
        print("                 \  (\_____/) /                  /o","                  \   UQDKR  (                  /o","                   \         (                ,o:",sep="\n")
        dumpseq_1 = dumpseq_1.fillna(-1)
        print("                   )          \,           .o;o'           ,o'o'o.","                 ./          /\o;o,,,,,;o;o;''         _,-o,-'''-o:o.","  .             ./o./)        \    'o'o'o''         _,-'o,o'         o",sep="\n")
        dumpseq_1.loc[:,'id_mpae'] = dumpseq_1.apply(lambda row: [int(row.module),int(row.path),int(row.activity),int(row.real_exercice)], axis=1)
        print("  o           ./o./ /       .o \.              __,-o o,o'","  \o.       ,/o /  /o/)     | o o'-..____,,-o'o o_o-'","  `o:o...-o,o-' ,o,/ |     \   'o.o_o_o_o,o--''",sep="\n")
        dumpseq_1.loc[:,'id_final_mpae'] = dumpseq_1.apply(lambda row: [int(row.real_module),int(row.real_path),int(row.real_activity),int(row.real_exercice)], axis=1)
        print("  .,  ``o-o'  ,.oo/   'o /\.o`.","  `o`o-....o'o,-'   /o /   \o \.                       ,o..         o","    ``o-o.o--      /o /      \o.o--..          ,,,o-o'o.--o:o:o,,..:o",sep="\n")
        diagnostique = dict_diagnostique(dumpseq_1)
        print("                  (oo(          `--o.o`o---o'o'o,o,-'''        o'o'o","                   \ o\              ``-o-o''''","    ,-o;o           \o \\","    ,-o;o           \o \\",sep="\n")
        
        # we have to transform the id_mpae column in string :
        dfnew = dumpseq_1[['id_mpae']].copy(deep=True)
        pds.options.mode.chained_assignment = None
        dumpseq_1['id_mpae'] = dfnew['id_mpae'].apply( lambda x : ''.join(str(e) for e in x) )
        dumpseq_1['diagnostique'] = dumpseq_1['id_mpae'].map(diagnostique)
        dumpseq_1['id_mpae'] = dfnew['id_mpae'].apply( lambda x : list( int(e) for e in x ) )

        print("   /o/               )o )","  (o(               /o /","   \o\.       ...-o'o /",sep="\n")
        dumpseq_1.to_pickle((os.path.join(".","data", f"sequences_data_{today}.pk1")))
        print("    \ o`o`-o'o o,o,--'","       ```o--'''",sep="\n")
        print("\nsequences dataframe built")

    if option == 2: #save 1 pandas dataframe : trackings

        print("\nbuilding sequences data\n")
        to_workable_json(original_json_sequences_name)
        json_to_csv_seq('jsonfile_to_csv_sequences2',f'transformed_{original_json_sequences_name}')
        print("                                              ,","                                             ,o","                                             :o",sep="\n")
        dumpseq_1 = csv_to_dataframe('jsonfile_to_csv_sequences2',['real_module','real_path','real_activity','real_exercice','difficultyseq', 'type_answer', 'module','path','activity','id_eleve', 'reponse_eleve', 'reponse_correcte'])
        print("                    _....._                  `:o","                  .'       ``-.                \o","                 /  _      _   \                \o",sep="\n")
        dumpseq_1[['real_module','real_path','real_activity','real_exercice']] = dumpseq_1[['real_module','real_path','real_activity','real_exercice']].apply(pds.to_numeric, errors='coerce')
        print("                :  /*\    /*\  :                 ;o","                |  \_/    \_/  :                  ;o","                (       U      /                  ;o",sep="\n")
        dumpseq_1[['real_module','real_path','real_activity','real_exercice']] = dumpseq_1[['real_module','real_path','real_activity','real_exercice']].astype('Int16')
        print("                 \  (\_____/) /                  /o","                  \   UQDKR  (                  /o","                   \         (                ,o:",sep="\n")
        dumpseq_1 = dumpseq_1.fillna(-1)
        print("                   )          \,           .o;o'           ,o'o'o.","                 ./          /\o;o,,,,,;o;o;''         _,-o,-'''-o:o.","  .             ./o./)        \    'o'o'o''         _,-'o,o'         o",sep="\n")
        dumpseq_1.loc[:,'id_mpae'] = dumpseq_1.apply(lambda row: [int(row.module),int(row.path),int(row.activity),int(row.real_exercice)], axis=1)
        print("  o           ./o./ /       .o \.              __,-o o,o'","  \o.       ,/o /  /o/)     | o o'-..____,,-o'o o_o-'","  `o:o...-o,o-' ,o,/ |     \   'o.o_o_o_o,o--''",sep="\n")
        dumpseq_1.loc[:,'id_final_mpae'] = dumpseq_1.apply(lambda row: [int(row.real_module),int(row.real_path),int(row.real_activity),int(row.real_exercice)], axis=1)
        print("  .,  ``o-o'  ,.oo/   'o /\.o`.","  `o`o-....o'o,-'   /o /   \o \.                       ,o..         o","    ``o-o.o--      /o /      \o.o--..          ,,,o-o'o.--o:o:o,,..:o",sep="\n")
        diagnostique = dict_diagnostique(dumpseq_1)
        print("                  (oo(          `--o.o`o---o'o'o,o,-'''        o'o'o","                   \ o\              ``-o-o''''","    ,-o;o           \o \\","    ,-o;o           \o \\",sep="\n")
        
        # we have to transform the id_mpae column in string :
        dfnew = dumpseq_1[['id_mpae']].copy(deep=True)
        pds.options.mode.chained_assignment = None
        dumpseq_1['id_mpae'] = dfnew['id_mpae'].apply( lambda x : ''.join(str(e) for e in x) )
        dumpseq_1['diagnostique'] = dumpseq_1['id_mpae'].map(diagnostique)
        dumpseq_1['id_mpae'] = dfnew['id_mpae'].apply( lambda x : list( int(e) for e in x ) )

        print("   /o/               )o )","  (o(               /o /","   \o\.       ...-o'o /",sep="\n")
        print("    \ o`o`-o'o o,o,--'","       ```o--'''",sep="\n")
        print("\nbuilding trackings data\n")
        to_workable_json(original_json_trackings_name)
        json_to_csv_track('jsonfile_to_csv_trackings2',f'transformed_{original_json_trackings_name}')
        print("                                                       ______","                                                     /    /_-._",sep="\n")
        print("                                                   / /_ ~~o\  :Y","                                                   / : \~x.  ` ')",sep="\n")
        dumptrack_2 = csv_to_dataframe('jsonfile_to_csv_trackings2',['token','module','path','activity','tms','exerciceId','isCorrect','question_type'])
        dico_etapes = remplir_etape()
        print("                                                  /  |  Y< ~-.__j")
        dumptrack_2['tms'] = dumptrack_2['tms']/1000
        dumptrack_2['jour'] = dumptrack_2.apply(lambda row: datetime.datetime.fromtimestamp(row.tms).weekday(), axis=1)
        print("                                                 UQDKR  l<  /.-~")
        dumptrack_2['etape'] = dumptrack_2.apply(lambda row: dico_etapes[(row.module,row.path,row.activity)],axis=1)
        print("                                                 ` l /~\ \<|Y")
        dumptrack_2['id_mpa'] = dumptrack_2.apply(lambda row: [int(row.module),int(row.path),int(row.activity)], axis=1)
        print("                                                    ',-~\ \L|")
        dumptrack_2['id_mpae'] = dumptrack_2.apply(lambda row: id_mpae_3_step(row), axis=1)
        print("'--.____  '--------.______       _.----.-----./      :/   '--'")
        dumptrack_2['date'] = dumptrack_2.apply(lambda row: (datetime.datetime.fromtimestamp(row.tms).year,datetime.datetime.fromtimestamp(row.tms).month,datetime.datetime.fromtimestamp(row.tms).day), axis=1)
        print("        '--.__            `'----/       '-.      __ :/")
        chatbot = set_essai(dumptrack_2)
        print("              '-.___           :           \   .'  )/")
        
        # we have to transform the id_mpae column in string :
        dfnew = chatbot[['id_mpae']].copy(deep=True)
        pds.options.mode.chained_assignment = None
        chatbot['id_mpae'] = dfnew['id_mpae'].apply( lambda x : ''.join(str(e) for e in x) )
        chatbot['diagnostique'] = chatbot['id_mpae'].map(diagnostique)
        chatbot['id_mpae'] = dfnew['id_mpae'].apply( lambda x : list( int(e) for e in x ) )

        chatbot.loc[chatbot['etape']!=1,'diagnostique'] = np.nan
        print("                    '---._           _.-'   ] /  _/")
        chatbot_new = duree(chatbot)
        print("                         '-._      _/     _/ / _/")
        chatbot_new = chatbot_new.sort_values('tms', axis=0, ascending=True)
        chatbot_new = pds.DataFrame.set_index(chatbot_new,np.arange(chatbot_new.shape[0]))
        print("                             \_ .-'____.-'__< |  \___")
        connection_index = get_index_connection(chatbot_new)
        chatbot_new['num_connection'] = chatbot_new.index.map(connection_index)
        print("                               <_______.\    \_\_---.7")
        came_back = {}
        for idt in chatbot_new['token'].unique() :
            came_back[idt] = sum(chatbot_new[chatbot_new['token']==idt]['duree'] > 2700)
        print("                              |   /'=r_.-'     _\\ =/")
        chatbot_new['reconnections'] = chatbot_new['token'].map(came_back)
        print("                          .--'   /            ._/'>")
        nbr_questions = {}
        for idt in chatbot_new['token'].unique() :
            nbr_questions[idt] = chatbot_new[chatbot_new['token']==idt].shape[0]
        print("                        .'   _.-'")
        chatbot_new['parcours_diagnostique'] = chatbot_new['diagnostique'].apply(get_parcours_diagnostique)
        chatbot_new['nbr_questions_faites'] = chatbot_new['token'].map(nbr_questions)
        print("                       / .--'")
        horaire = tranche_horaire(chatbot_new)
        print("                      /,/")
        chatbot_new['horaire'] = chatbot_new.apply( lambda row: horaire[(row.token,row.num_connection)],axis=1 )
        # WARNING : Pay attention of the current order of columns here befor changing the names :
        # print(chatbot_new.columns)
        chatbot_new.columns = ['id_eleve', 'module', 'path', 'activity', 'tps_posix', 'exercice', 'correct', 'question_type',
       'jour', 'etape', 'id_mpa', 'id_mpae', 'date', 'essai', 'diagnostique',
       'tps+1', 'duree', 'num_connection', 'reconnections', 
       'parcours_diagnostique', 'nbr_questions_faites', 'horaire']
        print("                      |/`)")
        # Now we can delete some unuseful columns 
        chatbot_final = chatbot_new.loc[:,['id_eleve', 'module', 'path', 'activity', 'tps_posix', 'exercice','correct', 'question_type',
       'jour', 'etape', 'id_mpa', 'id_mpae', 'date', 'essai',
       'tps+1', 'duree', 'num_connection', 'reconnections',
       'parcours_diagnostique', 'nbr_questions_faites', 'horaire']]
        print("                      'c=,")
        chatbot_final.to_pickle((os.path.join(".","data", f"chatbot_data_{today}.pk1")))
        print("\nchatbot dataframe built from trackings and sequences data\n")

def main(argv):
    if usage(argv) != 0:
        exit(0)
    if check_error(argv) != 0:
        exit(84)
    option = int(argv[1])
    seq_name = str(argv[2])
    track_name = str(argv[3])
    data_transformation(option, seq_name, track_name)
    return 0

if __name__ == '__main__':
    main(sys.argv)

