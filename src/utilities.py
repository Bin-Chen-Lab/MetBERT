import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
cache = set(stopwords.words("english"))

# loading words
ADMIN_LANGUAGE = [["FINAL REPORT",'']
    ,["Admission Date",'']
    ,["Discharge Date",'']
    ,["Date of Birth",'']
    ,["Phone",'']
    ,["Date/Time",'']
    ,["ID",'']
    ,["Completed by",'']
    ,["Dictated By",'']
    ,["Attending",'']
    ,["Provider: ",'']
    ,["Provider",'']
    ,["Primary",'']
    ,["Secondary",'']
    ,[" MD Phone",'']
    ,[" M.D. Phone",'']
    ,[" MD",'']
    ,[" PHD",'']
    ,[" X",'']
    ,[" IV",'']
    ,[" VI",'']
    ,[" III",'']
    ,[" II",'']
    ,[" VIII",'']
    ,["JOB#",'']
    ,["JOB#: cc",'']
    ,["# Code",'']
    ,["0.5 % Drops ",'']
    ,["   Status: Inpatient DOB",'']
    ,[" x",'']
    ,[" am",'']
    ,[" pm", '']
    ,["\n", " "]
    ,["\n\n", " "]
    ,["\n\n\n", " "]
    ,["\d+", ""]
    ,['\s+', ' ']
    ,['dr.','doctor']
    ,['q.d.', 'once a day']
    ,['b.i.d.', 'twice a day']
    ,['Subq.', 'subcutaneous']
    ,['q.i.d.', 'four times a day']
    ,['q.h.s.', 'before bed']
    ,['5x', 'a day five times a day']
    ,['q.4h', 'every four hours']
    ,['q4hours', 'every four hours']
    ,['q\.6h', 'every six hours']
    ,['q.o.d.', 'every other day']
    ,['prn\.', 'as needed']
    ,['p.r.n.', 'as needed']
    ,['h/o', 'history of']
    ,['w/', 'with']
    ,['s/p', 'status post']
    ,['w/o', 'without']
    ,['b.i.d.', 'twice a day']
    ,['t.i.d.', 'three times a day']
    ,['schatzki\'s', 'schatzkis']
    ,['everyd.', 'everyday']]

def preprocess_and_clean_notes(x):
    processed_text = x
    for find, replace in ADMIN_LANGUAGE:
        processed_text = re.sub(find, replace, processed_text)
    return processed_text

def remove_punctuation(x):
    no_punct = "".join([c for c in x if c not in string.punctuation])
    return no_punct

def testCachedSet(x):
    text = ' '.join([word for word in x.split() if word not in cache])
    return text

def lc_remove(x): 
    return ' '.join(word for word in x.split() if len(word)>=2)