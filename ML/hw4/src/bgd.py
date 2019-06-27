#!/usr/bin/env python
# coding: utf-8
"""

- Original Version

    Author: Susheel Suresh
    Last Modified: 04/03/2019

"""

import numpy as np
from classifier import BinaryClassifier
from utils import get_feature_vectors


class BGDHinge(BinaryClassifier):
    
    def __init__(self, args):
        # TO DO: Initialize parameters here
        self.b = 0
        self.gb = 0
        self.w = np.zeros(args.f_dim)
        self.gw = np.zeros(args.f_dim)
        self.vector_size = args.f_dim
        self.vocab_size = args.vocab_size
        self.bgd_learning_rate = args.lr_bgd
        self.iterate = args.num_iter
        self.binary = args.bin_feats
        self.lamb = float(args.__dict__['lambda']) if args.__dict__['lambda'] is not None else 0
        
    def fit(self, train_data):
        # TO DO: Learn the parameters from the training data
        data = get_feature_vectors(train_data[0], True)
        lab = train_data[1]
        for i in range(0, self.iterate):
            self.gw = np.zeros(self.vector_size)
            self.gb = 0
            for j in range(0, len(data)):
                original_value = lab[j]
                if original_value * (np.dot(self.w, np.array(data[j])) + self.b) <= 1:
                    self.gw += original_value * np.array(data[j])
                    self.gb += original_value
            self.b += self.bgd_learning_rate * self.gb
            self.w += self.bgd_learning_rate * self.gw
        
    def predict(self, test_x):
        # TO DO: Compute and return the output for the given test inputs
        data = get_feature_vectors(test_x, True)
        return [np.sign(np.dot(self.w, np.array(d)) + self.b) for d in data]


class BGDLog(BinaryClassifier):
    
    def __init__(self, args):
        # TO DO: Initialize parameters here
        self.b = 0
        self.gb = 0
        self.h = 0
        self.theta = 0.0
        self.w = np.zeros(args.f_dim)
        self.gw = np.zeros(args.f_dim)
        self.vector_size = args.f_dim
        self.vocab_size = args.vocab_size
        self.bgd_learning_rate = args.lr_bgd
        self.iterate = args.num_iter
        self.binary = args.bin_feats
        self.lamb = float(args.__dict__['lambda']) if args.__dict__['lambda'] is not None else 0

    def fit(self, train_data):
        # TO DO: Learn the parameters from the training data
        data = np.array(get_feature_vectors(train_data[0], True))
        lab = train_data[1]
        for i in range(0, self.iterate):
            self.gw = np.zeros(self.vector_size)
            self.gb = 0
            for j in range(0, data.shape[0]):
                pred = np.dot(self.w.T, data[j]) + self.b
                z = self.g(pred)
                temp = lab[j]
                if lab[j] < 0:
                    temp = 0
                self.gw += np.array(data[j]) * (temp - z)  # / float(data.shape[0])
                self.gb += lab[j] * (temp - z)  # / float(data.shape[0])
            self.w += self.bgd_learning_rate * np.array(self.gw)
            self.b += self.bgd_learning_rate * np.array(self.gb)

    def g(self, z):
        return 1.0 / (1.0 + np.exp(-1 * z))

    def predict(self, test_x):
        # TO DO: Compute and return the output for the given test inputs
        data = get_feature_vectors(test_x, True)
        return [np.sign(np.dot(self.w, np.array(d)) + self.b) for d in data]


class BGDHingeReg(BinaryClassifier):
    
    def __init__(self, args):
        # TO DO: Initialize parameters here
        self.b = 0
        self.gb = 0
        self.w = np.zeros(args.f_dim)
        self.gw = np.zeros(args.f_dim)
        self.vector_size = args.f_dim
        self.vocab_size = args.vocab_size
        self.bgd_learning_rate = args.lr_bgd
        self.iterate = args.num_iter
        self.binary = args.bin_feats
        self.lamb = float(args.__dict__['lambda']) if args.__dict__['lambda'] is not None else 0

    def fit(self, train_data):
        # TO DO: Learn the parameters from the training data
        data = get_feature_vectors(train_data[0], True)
        lab = train_data[1]
        for i in range(0, self.iterate):
            self.gw = np.zeros(self.vector_size)
            self.gb = 0
            for j in range(0, len(data)):
                original_value = lab[j]
                if original_value * (np.dot(self.w, np.array(data[j])) + self.b) <= 1:
                    self.gw += original_value * np.array(data[j])
                    self.gb += original_value
            self.gw -= self.lamb * self.w
            self.b += self.bgd_learning_rate * self.gb
            self.w += self.bgd_learning_rate * self.gw
        
    def predict(self, test_x):
        # TO DO: Compute and return the output for the given test inputs
        data = get_feature_vectors(test_x, True)
        return [np.sign(np.dot(self.w, np.array(d)) + self.b) for d in data]


class BGDLogReg(BinaryClassifier):

    def __init__(self, args):
        # TO DO: Initialize parameters here
        self.b = 0
        self.gb = 0
        self.h = 0
        self.theta = 0.0
        self.w = np.zeros(args.f_dim)
        self.gw = np.zeros(args.f_dim)
        self.vector_size = args.f_dim
        self.vocab_size = args.vocab_size
        self.bgd_learning_rate = args.lr_bgd
        self.iterate = args.num_iter
        self.binary = args.bin_feats
        self.lamb = float(args.__dict__['lambda']) if args.__dict__['lambda'] is not None else 0

    def fit(self, train_data):
        # TO DO: Learn the parameters from the training data
        data = np.array(get_feature_vectors(train_data[0], True))
        lab = train_data[1]
        for i in range(0, self.iterate):
            self.gw = np.zeros(self.vector_size)
            self.gb = 0
            for j in range(0, data.shape[0]):
                pred = np.dot(self.w.T, data[j]) + self.b
                z = self.g(pred)
                temp = lab[j]
                if lab[j] < 0:
                    temp = 0
                self.gw += np.array(data[j]) * (temp - z)  # / float(data.shape[0])
                self.gb += lab[j] * (temp - z)  # / float(data.shape[0])
            self.gw -= self.lamb * self.w
            self.w += self.bgd_learning_rate * np.array(self.gw)
            self.b += self.bgd_learning_rate * np.array(self.gb)

    def g(self, z):
        return 1.0 / (1.0 + np.exp(-1 * z))

    def predict(self, test_x):
        # TO DO: Compute and return the output for the given test inputs
        data = get_feature_vectors(test_x, True)
        return [np.sign(np.dot(self.w, np.array(d)) + self.b) for d in data]

