import pandas as pd
import numpy as np
from tqdm import tqdm

import nltk
from nltk.tokenize import word_tokenize
# nltk.download('punkt')

import time
import string
import itertools
import pickle

from collections import Counter
from itertools import filterfalse
from functools import reduce
from scipy import sparse

import gc

from dask import distributed
from dask.distributed import Client, LocalCluster




def main(vocabSize, tokenFile, outputFileName):

    #### CREATING VOCAB


    wordCount = pickle.load(open('wordCount','rb'))

    vocab = wordCount.most_common(vocabSize)

    ## Creating the Word-ID dictionaries
    id_to_word = {i:x[0] for i,x in enumerate(vocab)}

    word_to_id = {value:key for key,value in id_to_word.items()}

    wordSet = set(word_to_id.keys())



    #### DASK PROCESS

    client = Client()

    print(client)


    def createCMatrix(corpus):
        
        windowSize = 10

        cooccurrences = sparse.lil_matrix((vocabSize, vocabSize), dtype=np.float64)

        for doc in corpus:

            for center_index, center_word in enumerate(doc):
                
                if center_word not in wordSet:
                    continue
                
                context = doc[max(0, center_index - windowSize) : center_index]
                contextLen = len(context)
                
                

                for context_index, context_word in enumerate(context):

                    dist = contextLen - context_index

                    inc = 1.0/float(dist)
                    
                    if context_word in wordSet:

                        cooccurrences[word_to_id[center_word], word_to_id[context_word]] += inc                     
                        cooccurrences[word_to_id[context_word], word_to_id[center_word]] += inc                     

                        # center_id = word_to_id[center_word] 
                        # context_id = word_to_id[context_word] 

                        # if center_id<context_id:

                        #     cooccurrences[center_id, context_id] += inc                     
                        # else:
                        #     cooccurrences[context_id, center_id] += inc                     
        
        return cooccurrences
                
                
    def split(a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))




    corpus = pickle.load(open(tokenFile,'rb'))

        
    matrices = []

    print("Starting process")


    for sub in tqdm(split(corpus,10)):

        a = client.map(createCMatrix, list(split(sub,16)))
        b = client.gather(a)
        
        mat = reduce(lambda x,y : x+y, b)
        
        matrices.append(mat.copy())
        
        client.cancel(a)
        client.cancel(b)
        
        del a
        del b


    print("Creating Final Cooccurence matrix")

    finalMat = reduce(lambda x,y : x+y, matrices) 


    with open(outputFileName,'wb') as f:
        pickle.dump(finalMat,f)


    client.shutdown()




        
if __name__ == '__main__':


    #### USER INPUT

    vocabSize = 1000000

    for i in range(1):

        tokenFile = 'tokenizedData_'+str(i)

        outputFileName = 'cooccurMat_1'+str(i)

        main(vocabSize, tokenFile, outputFileName)
        
        
        
    
    