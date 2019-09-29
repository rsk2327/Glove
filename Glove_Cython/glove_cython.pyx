#!python
# distutils: language = c++
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np
import scipy.sparse as sp
import collections
from cython.parallel import parallel, prange

from libc.stdlib cimport malloc, free

from cython.operator cimport dereference as deref
from libcpp.vector cimport vector




cdef inline double double_min(double a, double b) nogil: return a if a <= b else b
cdef inline int int_min(int a, int b) nogil: return a if a <= b else b
cdef inline int int_max(int a, int b) nogil: return a if a > b else b


cdef extern from "math.h" nogil:
    double sqrt(double)
    double c_log "log"(double)


cdef int binary_search(int* vec, int size, int first, int last, int x) nogil:
    """
    Binary seach in an array of ints
    """

    cdef int mid

    while (first < last):
        mid = (first + last) / 2
        if (vec[mid] == x):
            return mid
        elif vec[mid] > x:
            last = mid - 1
        else:
            first = mid + 1

    if (first == size):
        return first
    elif vec[first] > x:
        return first
    else:
        return first + 1


cdef struct SparseRowMatrix:
    vector[vector[int]] *indices
    vector[vector[float]] *data


cdef SparseRowMatrix* new_matrix():
    """
    Allocate and initialize a new matrix
    """

    cdef SparseRowMatrix* mat

    mat = <SparseRowMatrix*>malloc(sizeof(SparseRowMatrix))

    if mat == NULL:
        raise MemoryError()

    mat.indices = new vector[vector[int]]()
    mat.data = new vector[vector[float]]()

    return mat


cdef void free_matrix(SparseRowMatrix* mat) nogil:
    """
    Deallocate the data of a matrix
    """

    cdef int i
    cdef int rows = mat.indices.size()

    for i in range(rows):
        deref(mat.indices)[i].clear()
        deref(mat.data)[i].clear()

    del mat.indices
    del mat.data

    free(mat)


cdef void increment_matrix(SparseRowMatrix* mat, int row, int col, float increment) nogil:
    """
    Increment the (row, col) entry of mat by increment.
    """

    cdef vector[int]* row_indices
    cdef vector[float]* row_data
    cdef int idx
    cdef int col_at_idx

    # Add new row if necessary
    while row >= mat.indices.size():
        mat.indices.push_back(vector[int]())
        mat.data.push_back(vector[float]())

    row_indices = &(deref(mat.indices)[row])
    row_data = &(deref(mat.data)[row])

    # Find the column element, or the position where
    # a new element should be inserted
    if row_indices.size() == 0:
        idx = 0
    else:
        idx = binary_search(&(deref(row_indices)[0]), row_indices.size(),
                            0, row_indices.size(), col)

    # Element to be added at the end
    if idx == row_indices.size():
        row_indices.insert(row_indices.begin() + idx, col)
        row_data.insert(row_data.begin() + idx, increment)
        return

    col_at_idx = deref(row_indices)[idx]

    if col_at_idx == col:
        # Element to be incremented
        deref(row_data)[idx] = deref(row_data)[idx] + increment
    else:
        # Element to be inserted
        row_indices.insert(row_indices.begin() + idx, col)
        row_data.insert(row_data.begin() + idx, increment)


cdef int matrix_nnz(SparseRowMatrix* mat) nogil:
    """
    Get the number of nonzero entries in mat
    """

    cdef int i
    cdef int size = 0

    for i in range(mat.indices.size()):
        size += deref(mat.indices)[i].size()

    return size


cdef matrix_to_coo(SparseRowMatrix* mat, int shape):
    """
    Convert to a shape by shape COO matrix.
    """

    cdef int i, j
    cdef int row
    cdef int col
    cdef int rows = mat.indices.size()
    cdef int no_collocations = matrix_nnz(mat)

    # Create the constituent numpy arrays.
    row_np = np.empty(no_collocations, dtype=np.int32)
    col_np = np.empty(no_collocations, dtype=np.int32)
    data_np = np.empty(no_collocations, dtype=np.float64)
    cdef int[:] row_view = row_np
    cdef int[:] col_view = col_np
    cdef double[:] data_view = data_np

    j = 0

    for row in range(rows):
        for i in range(deref(mat.indices)[row].size()):

            row_view[j] = row
            col_view[j] = deref(mat.indices)[row][i]
            data_view[j] = deref(mat.data)[row][i]

            j += 1

    # Create and return the matrix.
    return sp.coo_matrix((data_np, (row_np, col_np)),
                         shape=(shape,
                                shape),
                         dtype=np.float64)


cdef int words_to_ids(list words, vector[int]& word_ids,
                      dictionary, int supplied, int ignore_missing):
    """
    Convert a list of words into a vector of word ids, using either
    the supplied dictionary or by consructing a new one.

    If the dictionary was supplied, a word is missing from it,
    and we are not ignoring out-of-vocabulary (OOV) words, an
    error value of -1 is returned.

    If we have an OOV word and we do want to ignore them, we use
    a -1 placeholder for it in the word_ids vector to preserve
    correct context windows (otherwise words that are far apart
    with the full vocabulary could become close together with a
    filtered vocabulary).
    """

    cdef int word_id

    word_ids.resize(0)

    if supplied == 1:
        for word in words:
            # Raise an error if the word
            # is missing from the supplied
            # dictionary.
            word_id = dictionary.get(word, -1)
            if word_id == -1 and ignore_missing == 0:
                return -1

            word_ids.push_back(word_id)

    else:
        for word in words:
            word_id = dictionary.setdefault(word,
                                            len(dictionary))
            word_ids.push_back(word_id)

    return 0


def getGame(x, n):
    return x + 1


def fit_vectors(double[:, ::1] wordvec,
                double[:, ::1] wordvec_sum_gradients,
                double[::1] wordbias,
                double[::1] wordbias_sum_gradients,
                int[::1] row,
                int[::1] col,
                double[::1] counts,
                int[::1] shuffle_indices,
                double initial_learning_rate,
                double max_count,
                double alpha,
                double max_loss,
                int no_threads):
    """
    Estimate GloVe word embeddings given the cooccurrence matrix.
    Modifies the word vector and word bias array in-place.

    Training is performed via asynchronous stochastic gradient descent,
    using the AdaGrad per-coordinate learning rate.
    """

    # Get number of latent dimensions and
    # number of cooccurrences.
    cdef int dim = wordvec.shape[1]
    cdef int no_cooccurrences = row.shape[0]

    # Hold indices of current words and
    # the cooccurrence count.
    cdef int word_a, word_b
    cdef double count, learning_rate, gradient

    # Loss and gradient variables.
    cdef double prediction, entry_weight, loss

    # Iteration variables
    cdef int i, j, shuffle_index

    # We iterate over random indices to simulate
    # shuffling the cooccurrence matrix.
    with nogil:
        for j in prange(no_cooccurrences, num_threads=no_threads,
                        schedule='dynamic'):
            shuffle_index = shuffle_indices[j]
            word_a = row[shuffle_index]
            word_b = col[shuffle_index]
            count = counts[shuffle_index]

            # Get prediction
            prediction = 0.0

            for i in range(dim):
                prediction = prediction + wordvec[word_a, i] * wordvec[word_b, i]

            prediction = prediction + wordbias[word_a] + wordbias[word_b]

            # Compute loss and the example weight.
            entry_weight = double_min(1.0, (count / max_count)) ** alpha
            loss = entry_weight * (prediction - c_log(count))

            # Clip the loss for numerical stability.
            if loss < -max_loss:
                loss = -max_loss
            elif loss > max_loss:
                loss = max_loss

            # Update step: apply gradients and reproject
            # onto the unit sphere.
            for i in range(dim):

                learning_rate = initial_learning_rate / sqrt(wordvec_sum_gradients[word_a, i])
                gradient = loss * wordvec[word_b, i]
                wordvec[word_a, i] = (wordvec[word_a, i] - learning_rate 
                                      * gradient)
                wordvec_sum_gradients[word_a, i] += gradient ** 2

                learning_rate = initial_learning_rate / sqrt(wordvec_sum_gradients[word_b, i])
                gradient = loss * wordvec[word_a, i]
                wordvec[word_b, i] = (wordvec[word_b, i] - learning_rate
                                      * gradient)
                wordvec_sum_gradients[word_b, i] += gradient ** 2

            # Update word biases.
            learning_rate = initial_learning_rate / sqrt(wordbias_sum_gradients[word_a])
            wordbias[word_a] -= learning_rate * loss
            wordbias_sum_gradients[word_a] += loss ** 2

            learning_rate = initial_learning_rate / sqrt(wordbias_sum_gradients[word_b])
            wordbias[word_b] -= learning_rate * loss
            wordbias_sum_gradients[word_b] += loss ** 2    


def construct_cooccurrence_matrix(corpus, dictionary, int supplied,
                                  int window_size, int ignore_missing):
    """
    Construct the word-id dictionary and cooccurrence matrix for
    a given corpus, using a given window size.

    Returns the dictionary and a scipy.sparse COO cooccurrence matrix.
    """

    # Declare the cooccurrence map
    cdef SparseRowMatrix* matrix = new_matrix()

    # String processing variables.
    cdef list words
    cdef int i, j, outer_word, inner_word
    cdef int wordslen, window_stop, error
    cdef vector[int] word_ids

    # Pre-allocate some reasonable size
    # for the word ids vector.
    word_ids.reserve(1000)

    # Iterate over the corpus.
    for words in corpus:

        # Convert words to a numeric vector.
        error = words_to_ids(words, word_ids, dictionary,
                             supplied, ignore_missing)
        if error == -1:
            raise KeyError('Word missing from dictionary')

        wordslen = word_ids.size()

        # Record co-occurrences in a moving window.
        for i in range(wordslen):
            outer_word = word_ids[i]

            # Continue if we have an OOD token.
            if outer_word == -1:
                continue

            window_stop = int_min(i + window_size + 1, wordslen)

            for j in range(i, window_stop):
                inner_word = word_ids[j]

                if inner_word == -1:
                    continue

                # Do nothing if the words are the same.
                if inner_word == outer_word:
                    continue

                if inner_word < outer_word:
                    increment_matrix(matrix,
                                     inner_word,
                                     outer_word,
                                     1.0 / (j - i))
                else:
                    increment_matrix(matrix,
                                     outer_word,
                                     inner_word,
                                     1.0 / (j - i))

    # Create the matrix.
    mat = matrix_to_coo(matrix, len(dictionary))
    free_matrix(matrix)

    return mat






def transform_paragraph(double[:, ::1] wordvec,
                        double[::1] wordbias,
                        double[::1] paragraphvec,
                        double[::1] sum_gradients,
                        int[::1] row,
                        double[::1] counts,
                        int[::1] shuffle_indices,
                        double initial_learning_rate,
                        double max_count,
                        double alpha,
                        int epochs):
    """
    Compute a vector representation of a paragraph. This has
    the effect of making the paragraph vector close to words
    that occur in it. The representation should be more
    similar to words that occur in it multiple times, and
    less close to words that are common in the corpus (have
    large word bias values).

    This should be be similar to a tf-idf weighting.
    """

    # Get number of latent dimensions and
    # number of cooccurrences.
    cdef int dim = wordvec.shape[1]
    cdef int no_cooccurrences = row.shape[0]

    # Hold indices of current words and
    # the cooccurrence count.
    cdef int word_b, word_a
    cdef double count

    # Loss and gradient variables.
    cdef double prediction
    cdef double entry_weight
    cdef double loss
    cdef double gradient

    # Iteration variables
    cdef int epoch, i, j, shuffle_index

    # We iterate over random indices to simulate
    # shuffling the cooccurrence matrix.
    for epoch in range(epochs):
        for j in range(no_cooccurrences):
            shuffle_index = shuffle_indices[j]

            word_b = row[shuffle_index]
            count = counts[shuffle_index]

            # Get prediction
            prediction = 0.0
            for i in range(dim):
                prediction = prediction + paragraphvec[i] * wordvec[word_b, i]
            prediction += wordbias[word_b]

            # Compute loss and the example weight.
            entry_weight = double_min(1.0, (count / max_count)) ** alpha
            loss = entry_weight * (prediction - c_log(count))

            # Update step: apply gradients.
            for i in range(dim):
                learning_rate = initial_learning_rate / sqrt(sum_gradients[i])
                gradient = loss * wordvec[word_b, i]
                paragraphvec[i] = (paragraphvec[i] - learning_rate
                                   * gradient)
                sum_gradients[i] += gradient ** 2
