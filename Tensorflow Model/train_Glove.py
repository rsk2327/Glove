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
import tensorflow as tf

from collections import Counter
from itertools import filterfalse
from functools import reduce
from scipy import sparse

import gc

from dask import distributed
from dask.distributed import Client, LocalCluster






from tf_glove import *



####### READING DATA ##############

a = pickle.load(open('cooccurMat_0','rb'))

for i in range(1,10):
    b = pickle.load(open('cooccurMat_'+str(i),'rb'))
    a = a+b


context_array = pickle.load(open('context_embed_20','rb'))
focal_array = pickle.load(open('focal_embed_20','rb'))    

print("Data read")

####### TRAINING MODEL #################

model = GloVeModel(embedding_size=100, context_size=10,max_vocab_size=1000000, load_context_vecs =context_array , load_focal_vecs = focal_array )

model.fit_to_cmatrix(a, 'wordCount')

print("Training started")

for i in tqdm(range(10)):
	model.train(1000000)


model.saveEmbeddings(suffix = "_30")