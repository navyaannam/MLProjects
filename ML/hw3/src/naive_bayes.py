#!/usr/bin/env python
# coding: utf-8
"""© 2019 Rajkumar Pujari All Rights Reserved

- Original Version

    Author: Rajkumar Pujari
    Last Modified: 03/12/2019

"""

from collections import Counter

import numpy as np
from nltk.corpus import stopwords

from classifier import BinaryClassifier
from utils import get_feature_vectors, get_vocab


class NaiveBayes(BinaryClassifier):
    
    def __init__(self, args):
        # TO DO: Initialize parameters here
        self.vector_size = args.f_dim
        self.vocab_size = args.vocab_size
        self.learning_rate = args.lr
        self.iterate = args.num_iter
        self.binary = args.bin_feats
        self.pos_word = np.zeros(args.f_dim)
        self.neg_word = np.zeros(args.f_dim)
        self.pos_word_dne = np.zeros(args.f_dim)
        self.neg_word_dne = np.zeros(args.f_dim)
        self.all_word = np.zeros(args.f_dim)
        self.pos_prob = 1
        self.neg_prob = 1
        self.stop_words = set(stopwords.words('english'))

    def fit(self, train_data):
        # TO DO: Learn the parameters from the training data
        data = get_feature_vectors(train_data[0], binary=True)
        lab = train_data[1]

        c = Counter(lab)
        pos_count = c[1]
        neg_count = c[-1]

        total = pos_count + neg_count

        self.pos_prob = float(pos_count) / total
        self.neg_prob = float(neg_count) / total

        stop_indices = []
        current_vocab = get_vocab()
        for word in self.stop_words:
            if word in current_vocab:
                stop_indices.append(current_vocab[word])

        for i in range(0, len(data)):  # movie reviews
            for j in range(0, len(data[i])):  # dictionary
                if j not in stop_indices:
                    if lab[i] == 1:
                        self.pos_word[j] += data[i][j]
                    elif lab[i] == -1:
                        self.neg_word[j] += data[i][j]

        for i in range(0, len(self.pos_word)):
            self.pos_word[i] = (float(self.pos_word[i]) + 1) / (pos_count + self.vector_size)
        for i in range(0, len(self.neg_word)):
            self.neg_word[i] = (float(self.neg_word[i]) + 1) / (neg_count + self.vector_size)
        
    def predict(self, test_x):
        # TO DO: Compute and return the output for the given test inputs
        data = get_feature_vectors(test_x, binary=True)
        ret = np.zeros(len(test_x))
        for i in range(0, len(data)):
            pos_review = 0
            neg_review = 0
            for j in range(0, len(data[i])):
                if data[i][j] > 0:
                    if self.pos_word[j] > 0:
                        pos_review += np.log(self.pos_word[j])
                    if self.neg_word[j] > 0:
                        neg_review += np.log(self.neg_word[j])
                elif data[i][j] <= 0:
                    if self.pos_word_dne[j] > 0:
                        pos_review += np.log(self.pos_word_dne[j])
                    if self.neg_word_dne[j] > 0:
                        neg_review += np.log(self.neg_word_dne[j])

            pos_review += np.log(self.pos_prob)
            neg_review += np.log(self.neg_prob)

            if pos_review > neg_review:
                ret[i] = 1
            elif pos_review <= neg_review:
                ret[i] = -1
        return ret

