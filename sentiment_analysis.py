import sys
import math
import numpy as np
from collections import defaultdict

# window size for the CNN
width = 2

# number of filters
F = 100

# learning rate
alpha = 1e-1

# vocabsize: size of the total vocabulary
vocabsize = 10000

# vocab: the vocabulary dictionary with the word as key and its index as value
# the input will be transformed into numerical indices using the vocab dictionary
# as the input for the forward and backward algorithm
# e.g. if vocab = {'this': 0, 'is': 1, 'a': 2, 'short': 3, 'sentence': 4} and the training data is
# "this is a short sentence this is",
# the input to the forward and backward algorithm could be [0, 1, 2, 3, 4, 0, 1]
vocab = {}

np.random.seed(1)

# U and V are weight vectors of the hidden layer
# U: a matrix of weights of all inputs for the first
# hidden layer for all F filters in the
# where each filter has the size of vocabsize by width (window size)
# U[i, j, k] represents the weight of filter u_j
# for word with vocab[word] = i when the word is
# at the position k of the sliding window
#
U = np.random.normal(loc=0, scale=0.01, size=(vocabsize, F, width))

# V: the the weight vector of the F filter outputs (after max pooling)
# that will produce the output, i.e. o = sigmoid(V*h)
V = np.random.normal(loc=0, scale=0.01, size=(F))


# utility functions for layers of the network

def sigmoid(x):
    """
    helper function that computes the sigmoid function
    """
    return 1. / (1 + math.exp(-x))


def read_vocab(filename):
    """
    helper function that builds up the vocab dictionary for input transformation
    """
    file = open(filename)
    for line in file:
        cols = line.rstrip().split("\t")
        word = cols[0]
        idd = int(cols[1])
        vocab[word] = idd
    file.close()


def read_data(filename):
    """
    :param filename: the name of the file
    :return: list of tuple ([word index list], label)
    as input for the forward and backward function
    """
    data = []
    file = open(filename)
    for line in file:
        cols = line.rstrip().split("\t")
        label = int(cols[0])
        words = cols[1].split(" ")
        w_int = []
        for w in words:
            # skip the unknown words
            if w in vocab:
                w_int.append(vocab[w])
        data.append((w_int, label))
    file.close()
    return data


def train():
    """
    main caller function that reads in the names of the files
    and train the CNN to classify movie reviews
    """
    vocabFile = "vocab.txt"
    trainingFile = "movie_reviews.train"
    testFile = "movie_reviews.dev"

    read_vocab(vocabFile)
    training_data = read_data(trainingFile)
    test_data = read_data(testFile)

    for i in range(50):
        # confusion matrix showing the accuracy of the algorithm
        confusion_training = np.zeros((2, 2))
        confusion_validation = np.zeros((2, 2))

        for (data, label) in training_data:
            # do backward pass to update weights for both U and V
            backward(data, label)

            # do forward pass and evaluate
            prob = forward(data)["prob"]
            pred = 1 if prob > .5 else 0
            confusion_training[pred, label] += 1

        for (data, label) in test_data:
            # do forward pass and evaluate
            prob = forward(data)["prob"]
            pred = 1 if prob > .5 else 0
            confusion_validation[pred, label] += 1

        print("Epoch: {} \tDev accuracy: {:.3f}"
            .format(
            i,
            np.sum(np.diag(confusion_validation)) / np.sum(confusion_validation)))




# ## 1. Forward pass function

def forward(word_indices):
    """
    :param word_indices: a list of word indices, i.e. idx = vocab[word]
    :return: a result dictionary containing 3 items -
    result['prob']: output of the CNN algorithm.
    result['h']: the hidden layer output after max pooling, h = [h1, ..., hF]
    result['hid']: argmax of F filters, e.g. j of x_j
    e.g. for the ith filter u_i, tanh(word[hid[i], hid[i] + width]*u_i) = max(h_i)
    """

    h = np.zeros(F, dtype=float)
    hid = np.zeros(F, dtype=int)
    prob = 0.0

    # step 1. compute h and hid
    # loop through the input data of word indices and
    # keep track of the max filtered value h_i and its position index x_j
    # h_i = max(tanh(weighted sum of all words in a given window)) over all windows for u_i

    num_words = len(word_indices)

    for i in range(F):
        # pick out selected indices for every word
        selected_indices = U[word_indices, np.full(len(word_indices), i)]

        # initialize vector_sum to the first window value for each word
        vector_sum = selected_indices[np.arange(len(selected_indices) - width + 1),
                                      np.full(len(selected_indices) - width + 1, 0)]

        for k in range(1, width):
            # next vector is shifted k down from initial vector_sum
            next_vector = selected_indices[np.arange(k, len(word_indices) - width + 1 + k),
                                          np.full(len(selected_indices) - width + 1, k)]

            # add to total vector elementwise sum
            vector_sum = np.sum([vector_sum, next_vector], axis=0)

        h[i] = max(np.tanh(vector_sum))
        hid[i] = np.argmax(np.tanh(vector_sum))


    # step 2. compute probability
    # once h and hid are computed, compute the probability with sigmoid(h^TV)
    prob = sigmoid(np.dot(h,V))

    # step 3. return result
    return {"prob": prob, "h": h, "hid": hid}



# ## 2. Backward pass function

def backward(word_indices, true_label):
    """
    :param word_indices: a list of word indices, i.e. idx = vocab[word]
    :param true_label: true label (0, 1) of the movie reviews
    :return: None
    update weight matrix/vector U and V based on the loss function
    """
    global U, V
    pred = forward(word_indices)
    prob = pred["prob"]
    h = pred["h"]
    hid = pred["hid"]

    # update U and V here
    # loss_function = y * log(o) + (1 - y) * log(1 - o)
    #               = true_label * log(prob) + (1 - true_label) * log(1 - prob)
    # to update V: V_new = V_current + d(loss_function)/d(V)*alpha
    # to update U: U_new = U_current + d(loss_function)/d(U)*alpha

    first_part = (true_label - sigmoid(np.dot(V, h)))

    for i in range(F):

        d_total = first_part * V[i] * (1 - h[i]**2)
        for k in range(width):
            x_j = hid[i] + k
            U[word_indices[x_j], i, k] += d_total * alpha


    V = V + (true_label - sigmoid(np.dot(V,h))) * h * alpha



train()
