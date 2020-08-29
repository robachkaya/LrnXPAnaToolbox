import pandas as pds
import os

k = pds.read_pickle((os.path.join(".","data","chatbot_data_2020-08-29.pk1")))
q = pds.read_pickle((os.path.join(".","data","sequences_data_2020-08-29.pk1")))
pds.set_option('display.max_columns', None)

print(k)
print(q)