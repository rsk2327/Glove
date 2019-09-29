import numpy as np
import pandas as pd

import spacy
import gensim

from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
import matplotlib.pyplot as plt



import seaborn as sns 


def getHeatMap(model, wordList,vectorList = None):
    

    try:
    	vectorList = [model.get_vector(x).reshape(1,-1) for x in wordList]
    except:
    	vectorList = [model.word_vectors[model.dictionary[x]].reshape(1,-1) for x in wordList]



    if vectorList is None:
        vectorList = [model.get_vector(x).reshape(1,-1) for x in wordList]
    
    mat = []

    for i in range(len(wordList)):

        mat.append([])

        for j in range(len(wordList)):

            sim = cosine_similarity(vectorList[i], vectorList[j])
            mat[i].append(sim[0][0])
    
    df = pd.DataFrame(mat, columns=wordList, index=wordList)
    
    plt.figure(figsize = (10,10))
    sns.heatmap(df)

    

    
    




