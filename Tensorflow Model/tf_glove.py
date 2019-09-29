
from __future__ import division

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
import tensorflow as tf

from collections import Counter
from itertools import filterfalse
from functools import reduce
from scipy import sparse

import gc

from dask import distributed
from dask.distributed import Client, LocalCluster

from collections import Counter, defaultdict
import os
from random import shuffle
import tensorflow as tf


class NotTrainedError(Exception):
    pass

class NotFitToCorpusError(Exception):
    pass

class GloVeModel():
    def __init__(self, embedding_size, context_size, max_vocab_size=100000, min_occurrences=1,
                 scaling_factor=3/4, cooccurrence_cap=100, batch_size=512, learning_rate=0.05,
                load_context_vecs = None, load_focal_vecs = None):
        
        self.embedding_size = embedding_size
        if isinstance(context_size, tuple):
            self.left_context, self.right_context = context_size
        elif isinstance(context_size, int):
            self.left_context = self.right_context = context_size
        else:
            raise ValueError("`context_size` should be an int or a tuple of two ints")
        self.max_vocab_size = max_vocab_size
        self.min_occurrences = min_occurrences
        self.scaling_factor = scaling_factor
        self.cooccurrence_cap = cooccurrence_cap
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.__words = None
        self.__word_to_id = None
        self.__cmatrix = None
        self.__num_pairs = None
        
        
        self.__embeddings = None
        self.__focal_embeddings = None
        self.__context_embeddings = None
        self.__focal = None
        self.__context= None
        
        self.__load_context_vecs = load_context_vecs
        self.__load_focal_vecs = load_focal_vecs
        
        
        
    def fit_to_cmatrix(self, cmatrix, wordCountFile):
        """
        Fits a pre-build Cooccurence matrix to the model
        """
        print("In here")
        wordCount = pickle.load(open(wordCountFile,'rb'))
        vocab = wordCount.most_common(self.max_vocab_size)

        ## Creating the Word-ID dictionaries
        self.__id_to_word = {i:x[0] for i,x in enumerate(vocab)}

        self.__word_to_id = {value:key for key,value in self.__id_to_word.items()}
        
        self.__words = set(self.__word_to_id.keys())
        
        self.__cmatrix = cmatrix.tocoo()
        
        self.__num_pairs = len(self.__cmatrix.row)
        
        self.__build_graph()
        
           

    def __build_graph(self):
        self.__graph = tf.Graph()
        with self.__graph.as_default(), self.__graph.device("/device:GPU:0"):
            count_max = tf.constant([self.cooccurrence_cap], dtype=tf.float32,
                                    name='max_cooccurrence_count')
            scaling_factor = tf.constant([self.scaling_factor], dtype=tf.float32,
                                         name="scaling_factor")

            self.__focal_input = tf.placeholder(tf.int32, shape=[self.batch_size],
                                                name="focal_words")
            self.__context_input = tf.placeholder(tf.int32, shape=[self.batch_size],
                                                  name="context_words")
            self.__cooccurrence_count = tf.placeholder(tf.float32, shape=[self.batch_size],
                                                       name="cooccurrence_count")
            
            print("Right here")
            if self.__load_focal_vecs is None:
                focal_embeddings = tf.Variable(
                    tf.random_uniform([self.vocab_size, self.embedding_size], 1.0, -1.0),
                    name="focal_embeddings")
            else:
                print("Loading pretrained values")
                focal_embeddings = tf.Variable(self.__load_focal_vecs,name="focal_embeddings")
            
            if self.__load_context_vecs is None:
                context_embeddings = tf.Variable(
                    tf.random_uniform([self.vocab_size, self.embedding_size], 1.0, -1.0),
                    name="context_embeddings")
            else:
                print("Loading pretrained values")
                context_embeddings = tf.Variable(self.__load_context_vecs,name="context_embeddings")
                

            focal_biases = tf.Variable(tf.random_uniform([self.vocab_size], 1.0, -1.0),
                                       name='focal_biases')
            context_biases = tf.Variable(tf.random_uniform([self.vocab_size], 1.0, -1.0),
                                         name="context_biases")

            focal_embedding = tf.nn.embedding_lookup([focal_embeddings], self.__focal_input)
            context_embedding = tf.nn.embedding_lookup([context_embeddings], self.__context_input)
            focal_bias = tf.nn.embedding_lookup([focal_biases], self.__focal_input)
            context_bias = tf.nn.embedding_lookup([context_biases], self.__context_input)

            weighting_factor = tf.minimum(
                1.0,
                tf.pow(
                    tf.div(self.__cooccurrence_count, count_max),
                    scaling_factor))

            embedding_product = tf.reduce_sum(tf.multiply(focal_embedding, context_embedding), 1)

            log_cooccurrences = tf.log(tf.to_float(self.__cooccurrence_count))

            distance_expr = tf.square(tf.add_n([
                embedding_product,
                focal_bias,
                context_bias,
                tf.negative(log_cooccurrences)]))

            single_losses = tf.multiply(weighting_factor, distance_expr)
            self.__total_loss = tf.reduce_sum(single_losses)
            tf.summary.scalar("GloVe_loss", self.__total_loss)
            self.__optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(
                self.__total_loss)
            self.__summary = tf.summary.merge_all()

            self.__combined_embeddings = tf.add(focal_embeddings, context_embeddings,
                                                name="combined_embeddings")
            
            self.__focal_embeddings = focal_embeddings
            self.__context_embeddings = context_embeddings

    def train(self, num_steps = 1000, log_dir=None, summary_batch_interval=1000,
              tsne_epoch_interval=None):
       
        
        batches = self.getBatch()
        total_steps = 0
        
        with tf.Session(graph=self.__graph, config = tf.ConfigProto(allow_soft_placement = True)) as session:
            
            tf.global_variables_initializer().run()
            
            for step in range(num_steps):
                batch = self.getBatch()
                
                i_s, j_s, counts = zip(*batch)
                
                feed_dict = {
                        self.__focal_input: i_s,
                        self.__context_input: j_s,
                        self.__cooccurrence_count: counts}
                
                session.run([self.__optimizer], feed_dict=feed_dict)
            
            
                    
                total_steps += 1
                
            self.__embeddings = self.__combined_embeddings.eval()
            self.__focal  = self.__focal_embeddings.eval()
            self.__context = self.__context_embeddings.eval()
          
            
            
    def getBatch(self):
        
        batch = []
        
        for i in range(self.batch_size):
            ind = np.random.randint(self.__num_pairs)
            
            #Shuffling the center and context words because we have stored values only in one direction
            
            if np.random.random()>0.5:
                batch.append( (self.__cmatrix.row[ind], self.__cmatrix.col[ind], self.__cmatrix.data[ind]) )
            else:
                batch.append( (self.__cmatrix.col[ind], self.__cmatrix.row[ind], self.__cmatrix.data[ind]) )
            
            
        return batch
    
    def saveEmbeddings(self,suffix = ''):
        
        
        
        pickle.dump(self.__focal, open('focal_embed'+suffix,'wb'))
        pickle.dump(self.__context, open('context_embed'+suffix,'wb'))
            
            
            

    def embedding_for(self, word_str_or_id):
        if isinstance(word_str_or_id, str):
            return self.embeddings[self.__word_to_id[word_str_or_id]]
        elif isinstance(word_str_or_id, int):
            return self.embeddings[word_str_or_id]

    def __prepare_batches(self):
        if self.__cmatrix is None:
            raise NotFitToCorpusError(
                "Need to fit model to corpus before preparing training batches.")
        cooccurrences = [(word_ids[0], word_ids[1], count)
                         for word_ids, count in self.__cmatrix.items()]
        i_indices, j_indices, counts = zip(*cooccurrences)
        return list(_batchify(self.batch_size, i_indices, j_indices, counts))

    @property
    def vocab_size(self):
        return len(self.__words)

    @property
    def words(self):
        if self.__words is None:
            raise NotFitToCorpusError("Need to fit model to corpus before accessing words.")
        return self.__words

    @property
    def embeddings(self):
        if self.__embeddings is None:
            raise NotTrainedError("Need to train model before accessing embeddings")
        return self.__embeddings

    def id_for_word(self, word):
        if self.__word_to_id is None:
            raise NotFitToCorpusError("Need to fit model to corpus before looking up word ids.")
        return self.__word_to_id[word]

    def generate_tsne(self, path=None, size=(100, 100), word_count=1000, embeddings=None):
        if embeddings is None:
            embeddings = self.embeddings
        from sklearn.manifold import TSNE
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_embs = tsne.fit_transform(embeddings[:word_count, :])
        labels = self.words[:word_count]
        return _plot_with_labels(low_dim_embs, labels, path, size)


def _context_windows(region, left_size, right_size):
    for i, word in enumerate(region):
        start_index = i - left_size
        end_index = i + right_size
        left_context = _window(region, start_index, i - 1)
        right_context = _window(region, i + 1, end_index)
        yield (left_context, word, right_context)


def _window(region, start_index, end_index):
    """
    Returns the list of words starting from `start_index`, going to `end_index`
    taken from region. If `start_index` is a negative number, or if `end_index`
    is greater than the index of the last word in region, this function will pad
    its return value with `NULL_WORD`.
    """
    last_index = len(region) + 1
    selected_tokens = region[max(start_index, 0):min(end_index, last_index) + 1]
    return selected_tokens


def _device_for_node(n):
    if n.type == "MatMul":
        return "/gpu:0"
    else:
        return "/cpu:0"


def _batchify(batch_size, *sequences):
    for i in range(0, len(sequences[0]), batch_size):
        yield tuple(sequence[i:i+batch_size] for sequence in sequences)


def _plot_with_labels(low_dim_embs, labels, path, size):
    import matplotlib.pyplot as plt
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    figure = plt.figure(figsize=size)  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right',
                     va='bottom')
    if path is not None:
        figure.savefig(path)
        plt.close(figure)