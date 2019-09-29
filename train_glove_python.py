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
import time
import gc

from dask import distributed
from dask.distributed import Client, LocalCluster

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
plt.ioff()


import sys

sys.path.insert(0, '/home/santhosr/Documents/Courses/GloVe/glove_cython/Glove_Cython/')


## Importing Glove_Cython
import os
currDir = os.getcwd()
os.chdir("/home/santhosr/Documents/Courses/GloVe/glove_cython/Glove_Cython")
from glove_python import *
os.chdir(currDir)


## Importing Dictionary
vocabSize = 1000000

wordCount = pickle.load(open('wordCount','rb'))


vocab = wordCount.most_common(vocabSize)

id_to_word = {i:x[0] for i,x in enumerate(vocab)}

word_to_id = {value:key for key,value in id_to_word.items()}


## Importing WordList

wordList = []

with open('/home/santhosr/Documents/Courses/GloVe/financeWordList.txt') as f:
    wordList = f.readlines()

wordList = [x[:-1].split("\t") for x in wordList[1:]]

wordList = sorted(wordList, key=lambda x: x[1])

wordList = [x[0] for x in wordList]


a = pickle.load(open("/home/santhosr/Documents/Courses/GloVe/coo_full", 'rb'))
# a = pickle.load(open("/home/santhosr/Documents/Courses/GloVe/coo_full", 'rb'))


# word_to_id = {'is':0,'the':1, 'game':2,'time':4,'dfgf':5,'fgdfg':6,"dfgdfgdf":7,"qweqw":3}

corpus = Corpus(word_to_id)
corpus.matrix = a.tocoo()

glove = Glove(no_components=100, learning_rate=0.05)

# glove= Glove.load('glove_100_40iter.model')

glove.add_dictionary(corpus.dictionary)

# glove.loadStartIndex(40)

startTime = time.time()

glove.fit(corpus.matrix, epochs=150,
          no_threads=16, verbose=True, wordList = wordList,save_gap=1)

endTime = time.time()

print(endTime - startTime)

glove.save('glove_100_150iter.model')
