#!/usr/bin/env python
# coding: utf-8
"""© 2019 Rajkumar Pujari All Rights Reserved

- Original Version

    Author: Rajkumar Pujari
    Last Modified: 03/12/2019

"""
import numpy as np
from classifier import BinaryClassifier
from utils import get_feature_vectors


class Perceptron(BinaryClassifier):
    
    def __init__(self, args):
        # TO DO: Initialize parameters here
        self.b = 0
        self.w = np.zeros(args.f_dim)
        self.vector_size = args.f_dim
        self.vocab_size = args.vocab_size
        self.learning_rate = args.lr
        self.iterate = args.num_iter
        self.binary = args.bin_feats
        
    def fit(self, train_data):
        # TO DO: Learn the parameters from the training data
        data = get_feature_vectors(train_data[0], True)
        lab = train_data[1]

        for i in range(0, self.iterate):
            for j in range(0, len(data)):
                original_value = lab[j]
                sign_value = np.sign(np.dot(self.w, np.array(data[j])) + self.b)
                error = original_value - sign_value
                if error != 0:
                    self.b += error
                    shift = np.array([feat * error * self.learning_rate for feat in data[j]])
                    self.w = np.array(self.w) + shift
        
    def predict(self, test_x):
        # TO DO: Compute and return the output for the given test inputs
        data = get_feature_vectors(test_x, True)
        return [np.sign(np.dot(self.w, np.array(d)) + self.b) for d in data]


class AveragedPerceptron(BinaryClassifier):
    
    def __init__(self, args):
        # TO DO: Initialize parameters here
        self.b = 0
        self.beta = 0
        self.w = np.zeros(args.f_dim)
        self.u = np.zeros(args.f_dim)
        self.count = 1
        self.vector_size = args.f_dim
        self.vocab_size = args.vocab_size
        self.learning_rate = args.lr
        self.averaged_learning_rate = args.lra
        self.iterate = args.num_iter
        self.binary = args.bin_feats
                
    def fit(self, train_data):
        # TO DO: Learn the parameters from the training data
        data = get_feature_vectors(train_data[0], True)
        lab = train_data[1]
        count = 1

        for i in range(0, self.iterate):
            for j in range(0, len(data)):
                original_value = lab[j]
                sign_value = np.sign(np.dot(self.w, np.array(data[j])) + self.b)
                error = original_value - sign_value
                if error != 0:
                    self.b += error
                    self.w = np.array(self.w) + np.array(
                        [feat * error * self.averaged_learning_rate for feat in data[j]])
                    self.beta += error * count
                    self.u = np.array(self.u) + np.array(
                        [feat * error * count * self.averaged_learning_rate for feat in data[j]])
                count += 1
        self.b -= (self.beta/count)
        self.u = [feat/count for feat in self.u]
        self.w -= self.u

    def predict(self, test_x):
        # TO DO: Compute and return the output for the given test inputs
        data = get_feature_vectors(test_x, True)
        return [np.sign(np.dot(self.w, np.array(d)) + self.b) for d in data]


