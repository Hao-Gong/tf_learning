import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn


def generate_words_vector(training_set):
    words_vector = []
    for review in training_set:
        for word in review[0]:
            if word not in words_vector: words_vector.append(word)
    return words_vector


def generate_Y_vector(training_set, training_class):
    no_reviews = len(training_set)
    Y = np.zeros(shape=no_reviews)
    for ii in range(0, no_reviews):
        review_class = training_set[ii][1]
        Y[ii] = 1 if review_class == training_class else 0
    return Y


def generate_X_matrix(training_set, words_vector):
    no_reviews = len(training_set)
    no_words = len(words_vector)
    X = np.zeros(shape=(no_reviews, no_words + 1))
    for ii in range(0, no_reviews):
        X[ii][0] = 1
        review_text = training_set[ii][0]
        total_words_in_review = len(review_text)
        for word in set(review_text):
            word_occurences = review_text.count(word)
            word_index = words_vector.index(word) + 1
            X[ii][word_index] = word_occurences / float(total_words_in_review)
    return X

training_set = [([u'this', u'novel', u'if', u'bunch', u'of', u'childish', u'ladies'], 'neg'),
([u'where', u'to', u'begin', u'jeez', u'gasping', u'blushing',  u'fail????'], 'neg')]

print(training_set)

words_vector = generate_words_vector(training_set)
X = generate_X_matrix(training_set, words_vector)
Y_neg = generate_Y_vector(training_set, 'neg')


print(words_vector)
print(X)
print(Y_neg)