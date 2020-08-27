## EVIDENCEB, 2020
## library of function to use
## File description:
## functions to use in order to avoid pandas problem working with list 

def str_question_tolist(question):
    to_return = list( int(e) for e in question )
    if len(to_return)==5 :
        new_return = to_return[:3]
        new_return.append(int( str(to_return[3])+str(to_return[4]) ))
        return new_return
    else : 
        return to_return
    
def list_question_tostr(question):
    return ''.join(str(e) for e in question)

def from_list_to_str(df,column_name):
    # to avoid pandas error from manipulating lists in dataframe we convert question into string
    dfnew = df[[f"{column_name}"]].copy(deep=True)
    pds.options.mode.chained_assignment = None
    df[f"{column_name}"] = dfnew[f"{column_name}"].apply( lambda x : list_question_tostr(x) )
    return df

def from_str_to_list(df,column_name):
    # to get a better visualisation of questions at the end of the algorithm
    dfnew = df[[f"{column_name}"]].copy(deep=True)
    pds.options.mode.chained_assignment = None
    df[f"{column_name}"] = dfnew[f"{column_name}"].apply( lambda x : str_question_tolist(x) )
    return df
k = 'new_student_dataset_2020-08-26.pk1'
print(k.isidentifier())

