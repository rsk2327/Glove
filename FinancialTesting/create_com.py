import pandas as pd
import numpy as np
from tqdm import tqdm



import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

import time
import string
import itertools
import pickle

from collections import Counter
from itertools import filterfalse
from functools import reduce
from scipy import sparse
from tqdm import tqdm

import gc

from dask import distributed
from dask.distributed import Client, LocalCluster
import sys

sys.path.insert(0, '/home/santhosr/Documents/Courses/GloVe/glove_cython/Glove_Cython/')


## Importing Glove_Cython
import os
currDir = os.getcwd()
os.chdir("/home/santhosr/Documents/Courses/GloVe/glove_cython/Glove_Cython")
from glove_python import *
os.chdir(currDir)

####### Importing Dictionary ##############


word_to_id = pickle.load(open('/home/santhosr/Documents/Courses/GloVe/FinancialTesting/Data/comb_word_to_id','rb'))

id_to_word = pickle.load(open('/home/santhosr/Documents/Courses/GloVe/FinancialTesting/Data/comb_id_to_word','rb'))


for i in range(1,6):

	b = pickle.load(open('/home/santhosr/Documents/Courses/GloVe/FinancialTesting/Data/tokenizedData_'+str(i),'rb'))
	print("Loaded data")

	c = Corpus(word_to_id)

	c.fit(b, ignore_missing=True)

	mat = c.matrix.tocsr()

	with open('coo_'+str(i),'wb') as f:
		pickle.dump(mat,f)