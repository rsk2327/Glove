import pandas as pd
import numpy as np

import json

import os


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

import gc

from tqdm import tqdm

from dask import distributed
from dask.distributed import Client, LocalCluster

from glove import Corpus, Glove


data = pickle.load(open('tokenizedData','rb'))

corpus = Corpus()

corpus.fit(data)



# glove = Glove(no_components=200, learning_rate=0.05)
# glove.fit(corpus.matrix, epochs=20,
#           no_threads=8, verbose=True)
# glove.add_dictionary(corpus.dictionary)

# glove.save('glove_200_40iter.model')



## RETRAINING

glove = Glove.load('glove_200_40iter.model')

glove.fit(corpus.matrix, epochs=20,
          no_threads=8, verbose=True)

glove.save('glove_200_60iter.model')

glove.fit(corpus.matrix, epochs=20,
          no_threads=8, verbose=True)

glove.save('glove_200_80iter.model')
