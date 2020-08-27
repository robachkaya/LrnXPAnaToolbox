import pandas as pds
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

def recom_algorithm(userinput_df, students_df) :
    userinput_df = from_list_to_str(userinput_df,'question_id')
    students_df = from_list_to_str(students_df,'question_id')

# Beginning of the collaborative filtering.

# Filtering out students that have answered the same questions as the student X and storing it
    usersubset = students_df[students_df['question_id'].isin(userinput_df['question_id'])]
# Creating sub dataframes for every students
    usersubsetgroup = usersubset.groupby(['student_id'])
# Sorting it so students with questions most in common with the student X will have priority
    usersubsetgroup = sorted(usersubsetgroup, key=lambda x: len(x[1]), reverse=True)
# Creating the Pearson Correlation Coefficient dictionnary where the key is the student id and
# the value is the coefficient
    pearson_co = {}
# For every student group in our subdataframes
    for name, group in usersubsetgroup:
# Sort the student X and current student subdataframe so values are't mixed
        group = group.sort_values(by='question_id')
        userinput_df = userinput_df.sort_values(by='question_id')
# Get 'n' for pearson formula
        n = len(group)
# Get ratings that both students have in common in a temporary dataframe
        tmp_df = userinput_df[userinput_df['question_id'].isin(group['question_id'].tolist())]
# Then store it in a temporary dataframe
        tmp_rating = tmp_df['rating'].tolist()
# Put current student marks subdataframe in a list
        tmp_group = group['rating'].tolist()
# Now calculate Pearson Correlation between the two students (so called x & y)
        uxx = sum([i**2 for i in tmp_rating]) - pow(sum(tmp_rating), 2) / float(n)
        uyy = sum([i**2 for i in tmp_group]) - pow(sum(tmp_group), 2) / float(n)
        uxy = sum(i*j for i,j in zip(tmp_rating, tmp_group)) - sum(tmp_rating) * sum(tmp_group)/float(n)
# If the denominator is different than 0, divide, else there is no correlation (0)
        if uxx != 0 and uyy != 0:
            pearson_co[name] = uxy/sqrt(uxx*uyy)
        else:
            pearson_co[name] = 0
# Convert dictionnary into dataframe, adding column with the similarity index :
# Value vary from -1 to 1, where 1 is a direct correlation between the students, and -1
# is a negative correlation
    pearson_df = pds.DataFrame.from_dict(pearson_co, orient='index')
    pearson_df.columns = ['similarity_index']
    pearson_df['student_id'] = pearson_df.index
    pearson_df.index = range(len(pearson_df))
# Puts top 50 students that are the most similar to student X
    topstuds = pearson_df.sort_values(by='similarity_index', ascending=False)[0:50]

# Beggining of recommendation

# Merging similarity index dataframe and question_id dataframe
    topstudsrating = topstuds.merge(students_df, left_on='student_id', right_on='student_id', how='inner')
# Multiplying question rating by its weight (similarity index) then sum up the new ratings and divide it by
# the sum of the weights
    topstudsrating['weighted_rating'] = topstudsrating['similarity_index']*topstudsrating['rating']
# Applying a sum to the top students after grouping it up by students_id
    tmptopstudsrating = topstudsrating.groupby('question_id').sum()[['similarity_index', 'weighted_rating']]
    tmptopstudsrating.columns = ['sum_similarity_index','sum_weighted_rating']
# Create an empty dataframe and take weighted average
    recommendation_df = pds.DataFrame()
    recommendation_df['weighted_average_recom_score'] = tmptopstudsrating['sum_weighted_rating']/tmptopstudsrating['sum_similarity_index']
    recommendation_df['question_id'] = tmptopstudsrating.index
    recommendation_df = recommendation_df.sort_values(by='weighted_average_recom_score', ascending=False)
# Get dataframe of only questions id
    students = students_df.question_id.unique()
    questions_df = pds.DataFrame(students)
    questions_df.columns = ['question_id']
    recommendation_df = recommendation_df.reset_index(drop=True)
# Sort weighted average and see the top 20 questions the algorithm recommended
    recom = recommendation_df.loc[questions_df['question_id'].isin(recommendation_df['question_id'])]
    pds.set_option("display.max_rows", None, "display.max_columns", None)
    return recom.head(20)


# to avoid pandas error from manipulating lists in dataframe we convert question into string
#    dfnew = recommendation_dataset[['question_id']].copy(deep=True)
#    pds.options.mode.chained_assignment = None
#    recommendation_dataset['question_id'] = dfnew['question_id'].apply( lambda x : list_question_tostr(x) )
#    recommendation_dataset['question_id'] = dfnew['question_id'].apply( lambda x : str_question_tolist(x) )
    