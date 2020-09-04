=====================
UQDKR LrnXPAnaToolbox
=====================


.. image:: https://img.shields.io/pypi/v/LrnXPAnaToolbox.svg
        :target: https://pypi.python.org/pypi/LrnXPAnaToolbox

.. image:: https://img.shields.io/travis/robachkaya/LrnXPAnaToolbox.svg
        :target: https://travis-ci.com/robachkaya/LrnXPAnaToolbox

.. image:: https://readthedocs.org/projects/LrnXPAnaToolbox/badge/?version=latest
        :target: https://LrnXPAnaToolbox.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

TEST


Learning Experience Analysis Toolbox contains tools to analyse learning experience data using the pandas library. This package is developped for EvidenceB leraning experience data.


* Free software: MIT license
* Documentation: https://LrnXPAnaToolbox.readthedocs.io.



How to use
--------

## Installation

Run the command :

    pip install LrnXPAnaToolbox

Or if the package already exists and you just want to update it :

    pip install LrnXPAnaToolbox --upgrade


If you want to use this package in anaconda : go to the anaconda environment in which you are working 
(For example if you are working in a base environment run the following : )

    conda activat base

    pip install LrnXPAnaToolbox


If any problem prevent you to install the package you can :

* create a folder to clone the repository package

* in the folder :

    git clone https://github.com/robachkaya/LrnXPAnaToolbox

Or

    git clone git@github.com:robachkaya/LrnXPAnaToolbox.git

With the following key : uqdkr

Then

    python setup.py install

* go to the first folder name LrnXPAnaToolbox (where you can find a requirements_dev.txt)

    pip install -r requirements_dev.txt




LrnXPAnaToolbox functions
--------


LrnXPAnaToolbox

	transform_data
		transform_data.py		main function : data_transformation(option, original_json_sequences_name, original_json_trackings_name)
	
	create_marks
		create_marks.py			marks_table(pickle_file)
						conditionnal_proba(dataset, module, path, test, failure=False, get_proba_specific_path=False)
						complete_proba_matrix(proba, failure=False)

	cluster_students
		cluster_students.py		similar_students(available_database, new_student_data, clustering_plot=False)
						plot_clustering(DFVariables, DFMBKMEANS, nbr_clusters, nbr_components=3)
						creationDFClustering(available_database)
						optimal_n_clusters(data, clustering_method, nbr_students)


	recommender
		dropout.py			dropout_recommendation(new_student_data, available_database, recommendation_dataset)
						dropout_prediction_training_data(dataset, module_concerned, path_concerned, dropout_after_activity)
						dropout_prediction(prediction_test, student_array_features)

		final_test_recommendation.py	recom_algorithm(userinput_df, students_df)

		recommender.py			main function : algorithm(student_data, student_marks, students_df, marks_df)


* transform_data() take the names of the json files of data collected by the EvidenceB developpers, the objective of this function is to transform the pickle files (with the name passed as parameters) into pandas dataframe table for the data analysts.
The option is here to specify which file you want to transform :
option = 3       Transform data and save 2 pandas dataframes : sequences and chatbot
option = 1       Transform data and save 1 pandas dataframe : sequences
option = 2       Transform data and save 1 pandas dataframe : chatbot

sequences is usefull if you want to know about what is shown on the chatbot (the messages, the questions...).

chatbot (is mainly computed from the trackings file) is usefull to manipulate the data, you can implement new function in order to get new variables.


* marks_table() returns a dataframe with 3 columns the student, the question and the mark. This mark used to represent the learning achieved by the student with the exercise. More the mark has a high value, more the question is useful for the learning of the student. One of the tenets of this function is that : making mistakes is useful.
This function is useful to create the marks dataframe, needed to compute the LrnXPAnaToolbox.recommender.recommender.argorithm() function later. The only parameter for this function is the name of the pickle file of data (you can get one pickle like this one with LrnXPAnaToolbox.transform_data()).


* cluster_students.similar_students() fucntion create a dataframe and a predicition cluster value. Given a new student dataset + the dataset of all available data, this function returns a dataframe with the clusters for each student of the available data (using MiniBatchKMeans algorithm from sklearn) and the predicted cluster for the new student.
The dataset of available data could be obtain from the pickle file get with LrnXPAnaToolbox.transform_data().
The student dataset too.


* recommender main function is algorithm(), this function use cluster_students.similar_students(), recommender.recom_algorithm() and recommender.dropout_recommendation() to recommend questions given a new student dataset and his/her table marks + the available data dataset and the table of marks corresponding.


* the main function in final_recommendation() is recom_algorithm() which take the new student data and his/her table of marks. This function returns a dataframe with some best questions to propose to the student.


* the main function in dropout() called dropout_recommendation() modify the recommendation dataframe that you can get with...
As it is the case for the cluster_students function this function takes a new student dataset and the dataset of available data + the recommendation dataframe.




Example of use (for a data analyst)
--------

LrnXPAnaToolbox.transform_data.transform_data() will take the json files of the developpers and an option (with option = 3, for example, you will get the max of this function) as parameters. 
From the chatbot pickle created you can compute marks to get a big table of marks for each students and questions. 
Then when you collect the data of a new student on the chatbot you can do the same thing : transform the data and create the marks (this will be way faster compared to the time spent to compute the dataframe for all data).
To give you an idea, computing the transform data fucntion on all data take something like 20 minutes when it takes ................ for only one student.
Computing the create marks function on all data will take 3 hours comparing to a few minutesfor only one student.
The objective doing this is to recommend question(s) to the new student for his/her next connection. 
To do so, given the forth computed dataframes you can recommend question with the recommender function.

To transform data and for the futur manipulations : 
You have to create a data file in which you will save the original json data files from the developpers.
During the transformation from json files to pickle files which take place executing the LrnXPAnaToolbox.transform_data() function some new files will appear in this data folder.
Then it is in this data folder that you will able to retrieve the pickle file created.




Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
