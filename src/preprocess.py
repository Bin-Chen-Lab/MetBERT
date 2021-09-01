import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from utilities import *

# read data
final = pd.read_csv('../data/final.csv')

# Remove everything leading up till 'Chief Complaint'
final['TEXT'] = final['TEXT'].apply(lambda x: x.rpartition('CHIEF COMPLAINT:')[-1])

# removing deidentifed brackets
final["TEXT"] = final["TEXT"].replace(r"\[.*?\]", "", regex=True)

# removing medical jargons
final['TEXT'] = final['TEXT'].apply(preprocess_and_clean_notes)

final["TEXT"] = final["TEXT"].str.strip()
final['TEXT'] = final['TEXT'].str.lower()

# remve punctuations
final['TEXT'] = final['TEXT'].apply(lambda x: remove_punctuation(x), 1)

# removing stopwords is optional, not remvoing them actaully showed better performance in our case
final['TEXT'] = final['TEXT'].apply(lambda x: testCachedSet(x))

# removing all single and double letter words
final['TEXT'] = final['TEXT'].apply(lambda x: lc_remove(x))

final.insert(0, 'TEXT', final.pop("TEXT"))
final['TEXT'] = final['TEXT'].apply(lambda x: " ".join(x.split()))

# prepping for train-test-split
# selecting advanced cancer phenotype , you can select any on the phenotype you want to explore here
y = final['Advanced.Cancer']
X = final['TEXT']

# As we have imbalanced dataset, use stratify option
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0, stratify = y, shuffle=True)
# X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=0, stratify=y_test, shuffle=True)

# Merge them to get the final sets
# You can uncomment to create valdiation set
dtrain = pd.concat([X_train, y_train], axis=1)
#dval = pd.concat([X_val, y_val], axis=1)
dtest = pd.concat([X_test, y_test], axis=1)

dtrain.columns = ['TEXT', 'LABEL']
# dval.columns = ['TEXT', 'LABEL']
dtest.columns = ['TEXT', 'LABEL']

# save as csv
dtrain.to_csv('../data/can_train.csv', index = False)
# dval.to_csv('../data/can_val.csv', index = False)
dtest.to_csv('../data/can_test.csv', index = False)