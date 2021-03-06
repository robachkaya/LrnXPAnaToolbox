## EVIDENCEB, 2020
## cluster_students.py
## File description:
## cluster students function

import pandas as pds
import numpy as np
import os
import datetime
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn import metrics
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys

def usage(argv): # argv = arguments passed to the script = cluster_students.py -h
    if len(argv) == 2:
        if argv[1] == "-h":
            print("\nUSAGE")
            print("    ./cluster_students.py  available_database  new_student_data  clustering_plot")
            print("\nDESCRIPTION")
            print("""    available_database       dataframe with all the data (created with transform_data.py for example)""")
            print("""    new_student_data       dataframe with all the data for only one (new) student (created with transform_data.py for example)""")
            print("""    clustering_plot       default value set to False, if True the function plot the 3D PCA graph of the clustering""")
            return 1
    return 0

def check_error(argv): # argv = arguments passed to the script = cluster_students.py available_database new_student_data clustering_plot
    if (len(argv) != 4) & (len(argv) != 3) :
        print("Wrong number of arguments, try \"-h\" for more informations")
        return 1
    if len(argv) == 4 :
        if argv[3] not in ['True','False'] :
            print("Argument have to be True if you want to see the graphic representation of the clustering or False if not.")
            return 1
        else :
            return 0
    elif len(argv) != 3 :
        return 1
    else :
        return 0
    #if (len(argv) != 4) & (len(argv) != 3) :
    #    print("Wrong number of arguments, try \"-h\" for more informations")
    #    return 1
    #if len(argv) == 4 :
    #    if argv[3] not in ['True','False'] :
    #        print("Argument have to be True if you want to see the graphic representation of the clustering or False if not.")
    #        return 1
    #    if (argv[2].isidentifier() != True)|(argv[1].isidentifier() != True) :
    #        print("available_database and new_student_data arguments have to be the name of pandas dataframe so : alphanumeric letters (a-z) and/or (0-9) and/or underscores (_).")
    #        return 1
    #elif len(argv) == 3 :
    #    if (argv[2].isidentifier() != True)|(argv[1].isidentifier() != True) :
    #        print("available_database and new_student_data arguments have to be the name of pandas dataframe so : alphanumeric letters (a-z) and/or (0-9) and/or underscores (_).")
    #        return 1
    #else :
    #    return 1
    #return 0

def plot_clustering(DFVariables, DFMBKMEANS, nbr_clusters, nbr_components=3) :
    """Show and save in the data folder the figure of a clustering given with the DFMBKMEANS table."""
    # standardisation
    featuresPCA = DFVariables.copy()
    featuresPCA = StandardScaler().fit_transform(featuresPCA)
    pca = PCA(svd_solver='full')
    pca.fit_transform(featuresPCA)
    print(f"\nWith 3 principal components we explain {round((pca.explained_variance_ratio_[0]+pca.explained_variance_ratio_[1]+pca.explained_variance_ratio_[2])*100,2)}% of the total information,\nthe default number of principal components is set to 3.")

    n = DFVariables.shape[0] # observations
    p = DFVariables.shape[1] # variables

    # SCREE PLOT
    # corrected explained variance
    # var = (n-1)/n*pca.explained_variance_
    # threshold for the scree plot
    # bs = 1/np.arange(p,0,-1)
    # bs = np.cumsum(bs)
    # bs = bs[::-1]
    # scree plot values : we want eigenvalue > threshold
    # print(pds.DataFrame({'eigenvalue':var,'threshold':bs}))

    # 3D PCA Projection :
    pca = PCA(n_components=nbr_components)
    principalComponents = pca.fit_transform(featuresPCA)
    principalDf = pds.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2', 'principal component 3'])
    finalDf = pds.concat([principalDf, DFMBKMEANS[['Cluster']]], axis = 1)
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(25,-120)
    xAxisLine = ((min(principalComponents[:,0]), max(principalComponents[:,0])), (0, 0), (0,0))
    yAxisLine = ((0, 0), (min(principalComponents[:,1]), max(principalComponents[:,1])), (0,0))
    zAxisLine = ((0, 0), (0,0), (min(principalComponents[:,2]), max(principalComponents[:,2])))
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_zlabel('Principal Component 3', fontsize = 15)
    ax.set_title('3 Components PCA of the clustering', fontsize = 20)
    targets = np.arange(nbr_clusters)
    colors_to_chose = ['peachpuff','lightseagreen','peru','b','g','r', 'c','m', 'y', 'k','sandybrown','seashell','turquoise','aquamarine','lightcyan']
    colors = colors_to_chose[:nbr_clusters]
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['Cluster'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , finalDf.loc[indicesToKeep, 'principal component 3']
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.show()
    today = datetime.date.today()
    plt.savefig( (os.path.join(".","data", f"MBKMeans_figure_{today}.png")) )

def creationDFClustering(available_database):
    """Create the individuals representation vectors with some features."""
    d = {}
    grouped = available_database[(available_database.num_connection==1)]
    filtered = grouped.groupby('id_eleve')
    for student, student_groupby in filtered :

        # we first add the student
        d[student] = [student]

        student_data = available_database[available_database.id_eleve == student]

        # we add the average response time
        d[student].append( sum(student_data.duree)/(student_data.shape[0]) )

        # we add the success rate
        d[student].append( student_data[student_data.correct == True].shape[0]/student_data.shape[0] )

        # we add the path assigned to the student
        path_data = student_data[student_data.etape == 2]
        if path_data.shape[0] == 0 :
            d[student].append(0)
        else:
            path_data.sort_values(by = 'tps_posix', ascending = True)
            d[student].append( path_data.iloc[0]['path'] )

        # we add the total time spent on the chatbot
        d[student].append( sum(student_data.duree) )

        # we add the number of questions the student tried
        d[student].append( student_data.shape[0] )

        # we add the day
        day = student_data.iloc[0]['jour']
        d[student].append( day )

        # we add the timeslot
        timeslot = student_data.iloc[0]['horaire']
        d[student].append( timeslot )

    df = pds.DataFrame(np.array(list(d.values())))
    df.columns = ['id_eleve','temps_moyen_reponse','taux_reussite','parcours','duree_totale','nbr_questions','jour','horaire']
    return df

def optimal_n_clusters(data, clustering_method, nbr_students):
    """ Choose the optimal number (between 3 and 30) of cluster to use according to some criteria :
        - the minimal population of the clustering (so the number of individuals in the smallest cluster) must be >= to 1/50*the total number of individuals
          for example if there are 2500 students in the data, the minimal population must be >= 50 individuals
        - the maximal population of the clustering (so the number of individuals in the biggest cluster) must be <= to 1/2.5*the total number of individuals
          for example if there are 2500 students in the data, the maximal population must be <= 1000 individuals
    """
    optimal = False
    while optimal == False :
        choice = np.array([])
        population_min = np.array([])
        for k in range(28): # k : 0+3 > 27+3
            cls = clustering_method(n_clusters=k+3)
            cls.fit(data)
            groups = cls.labels_
            unique, counts = np.unique(groups, return_counts=True)
            dic = dict(zip(unique, counts))
            if (min(dic.values()) >= 1/50*nbr_students) & (max(dic.values()) <= 1/2.5*nbr_students) :
                choice = np.append(choice, metrics.silhouette_score(data, groups))
            else :
                choice = np.append(choice, -1)
        if np.mean( choice ) != -1 :
            optimal = True
    return np.flip(np.argsort(choice)[-3:])+3

def similar_students(available_database, new_student_data, clustering_plot=False) :
    """ Return 2 elements :
        - a table with 2 columns : students, clusters
        - the predicted cluster for an array of data (array of the data for one student which is not supposed to be in the data).
    
        Save the MiniBatchKMeans dataframe to pickle in the data folder
    """
    student_array_features = np.array([])
    # we add the average response time
    student_array_features = np.append( student_array_features, sum(new_student_data.duree)/(new_student_data.shape[0]) )
    # we add the success rate
    student_array_features = np.append( student_array_features,new_student_data[new_student_data.correct == True].shape[0]/new_student_data.shape[0] )
    # we add the path assigned to the student
    # WARNING : if the student has done 2 modules at his/her first connection we'll have 2 assigned path, which one to choose?
    path_data = new_student_data[new_student_data.etape == 2]
    if path_data.shape[0] == 0 :
        student_array_features = np.append( student_array_features,0 )
    else:
        path_data.sort_values(by = 'tps_posix', ascending = True)
        student_array_features = np.append( student_array_features,path_data.iloc[0]['path'] )
    # we add the total time spent on the chatbot
    student_array_features = np.append( student_array_features,sum(new_student_data.duree) )
    # we add the number of questions the student tried
    student_array_features = np.append( student_array_features,new_student_data.shape[0] )
    # we add the day
    day = new_student_data.iloc[0]['jour']
    student_array_features = np.append( student_array_features,day )
    # we add the timeslot
    timeslot = new_student_data.iloc[0]['horaire']
    student_array_features = np.append( student_array_features,timeslot )
    student_array_features = np.array([student_array_features])

    clustering_method = MiniBatchKMeans # the MiniBatchKMeans is the best method for our data.
    X = creationDFClustering(available_database)
    nbr_students = len(X['id_eleve'].unique())
    DFVariables = X.drop(['id_eleve'], axis = 1)
    DFVariables = DFVariables.apply(pds.to_numeric)
    normalized_data = preprocessing.normalize(DFVariables)
    nbr_clusters = optimal_n_clusters(normalized_data, clustering_method, nbr_students)[0]
    clustering_model = clustering_method(n_clusters=nbr_clusters)
    clustering = clustering_model.fit(normalized_data)
    DFMBKMEANS = X.copy()
    DFMBKMEANS['Cluster'] = clustering.labels_
    today = datetime.date.today()
    DFMBKMEANS.to_pickle((os.path.join(".","data", f"DF_MBKMeans_{today}.pk1")))
    if str(clustering_plot) == 'True' :
        print('\nPlease close the figure if ou want to continue.\n')
        plot_clustering(DFVariables, DFMBKMEANS, nbr_clusters)
    return DFMBKMEANS.loc[:,['id_eleve','Cluster']], clustering_model.predict(student_array_features)

def main(argv):
    if usage(argv) != 0:
        exit(0)
    if check_error(argv) != 0:
        exit(84)
    available_database = argv[1]
    new_student_data = argv[2]
    if len(argv)==4:
        clustering_plot = argv[3]
        similar_students(available_database, new_student_data, clustering_plot)
    elif len(argv)==3:
        similar_students(available_database, new_student_data)
    else:
        print("Wrong number of arguments, try \"-h\" for more informations")
    return 0

if __name__ == '__main__':
    main(sys.argv)