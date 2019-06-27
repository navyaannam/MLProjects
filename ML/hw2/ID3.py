##############
# Name: Navya Annam
# email: nannam@purdue.edu
# Date: 2/24/19

# need to do validation set

import sys
import pandas as pd
import os
import numpy as np
import math
import matplotlib.pyplot as mat


def entropy(freqs):
    all_freq = sum(freqs)
    if all_freq == 0:
        return 0
    entropy = 0
    for fq in freqs:
        prob = fq * 1.0 / all_freq
        if abs(prob) > 1e-8:
            entropy += -prob * np.log2(prob)
    return entropy


def infor_gain(before_split_freqs, after_split_freqs):
    gain = entropy(before_split_freqs)
    overall_size = sum(before_split_freqs)
    for freq in after_split_freqs:
        ratio = sum(freq) * 1.0 / overall_size
        gain -= ratio * entropy(freq)
    return gain


class Node(object):
    def __init__(self, leaf=False, cls=None, parent=None):
        self.no = None
        self.yes = None
        self.parent = parent
        self.value = None
        self.feature = None
        self.continuous = None
        self.leaf = leaf
        self.cls = cls # dead or alive

    def __str__(self):
        if self.leaf:
            return "[Leaf] Class: {}".format(self.cls)
        else:
            return "feature: {}, value: {}, continuous: {}".format(self.feature, self.value, self.continuous)


class Tree(object):

    def __init__(self, model_type, train_size, max_depth=None, sample_split_size=None):
        self.train_size = train_size
        self.model_type = model_type
        self.max_depth = max_depth
        self.min_sample_split_size = sample_split_size

        self.leaves = []
        self.non_leaves = []
        self.continuous_attributes = [2, 3, 5]

        self.root = None

        self.lowest_level = 0

        self.vanilla_test_set_accuracy = [0.8209, 0.7985, 0.8134, 0.8060]
        self.vanilla_training_set_accuracy = [0.8795, 0.8633, 0.8695, 0.8700]
        self.vanilla_training_set_percentage = [.4, .6, .8, 1.0]
        self.node_count = 0
        self.nodes = [287, 287, 287, 287, 287]

        self.depth_test_set_accuracy_five = [0.8284, 0.8284, 0.8284, 0.8284, 0.8284]
        self.depth_validation_set_accuracy_five = [0.7840, 0.7949, 0.7861, 0.7918, 0.7996]
        self.depth_training_set_accuracy_five = [0.8434, 0.8232, 0.8257, 0.8165, 0.8153]

        self.depth_test_set_accuracy_ten = [0.8097, 0.8097, 0.8097, 0.8097, 0.8097]
        self.depth_validation_set_accuracy_ten = [0.8600, 0.8622, 0.8476, 0.8535, 0.8577]
        self.depth_training_set_accuracy_ten = [0.8956, 0.8714, 0.8713, 0.8647, 0.8695]

        self.depth_test_set_accuracy_fifteen = [0.8060, 0.8060, 0.8060, 0.8060, 0.8060]
        self.depth_validation_set_accuracy_fifteen = [0.8720, 0.8718, 0.8610, 0.8604, 0.8637]
        self.depth_training_set_accuracy_fifteen = [0.8835, 0.8682, 0.8686, 0.8647, 0.8695]

        self.depth_test_set_accuracy_twenty = [0.8060, 0.8060, 0.8060, 0.8060, 0.8060]
        self.depth_validation_set_accuracy_twenty = [0.8720, 0.8718, 0.8610, 0.8604, 0.8637]
        self.depth_training_set_accuracy_twenty = [0.8835, 0.8682, 0.8686, 0.8647, 0.8695]

        self.depth_test_set_accuracy_max = [0.8060, 0.8060, 0.8060, 0.8060, 0.8060]  # choose values vertically
        self.depth_validation_set_accuracy_max = [0.8720, 0.8718, 0.8610, 0.8604, 0.8637]
        self.depth_training_set_accuracy_max = [0.8835, 0.8682, 0.8686, 0.8647, 0.8695]

        self.depth_depth_max = [20, 20, 20, 20, 20]

        self.depth_training_set_percentage = [.4, .5, .6, .7, .8]

        self.prune_test_set_accuracy = [0.8022, 0.8097, 0.8097, 0.8097, 0.8097]
        self.prune_training_set_accuracy = [0.8755, 0.8650, 0.8713, 0.8647, 0.8715]
        self.prune_training_set_percentage = [.4, .5, .6, .7, .8]

    def build_tree(self, data, split_candidates=None, level=0):
        # data contains the subset after every split
        # pairs of attributes and values where it has already split

        self.lowest_level = level if level > self.lowest_level else self.lowest_level

        if split_candidates is None:
            split_candidates = []
            for i in range(7):
                values = np.unique(data[:, i])  # get unique values of that column
                for value in values:
                    split_candidates.append((value, i))  # append each value and column number
        else:
            split_candidates = split_candidates.copy()  # copy so doesn't change previous arrays

        if self.model_type == "min_split" and data.shape[0] <= self.min_sample_split_size:
            vals, counts = np.unique(data, return_counts=True)
            node = Node(True, vals[np.argmax(counts)])
            self.node_count += 1
            self.leaves.append(node)
            return node
        if data.shape[0] == 1 or np.unique(data).shape[0] == 1:  # if there is only one node, or only 1 value in column
            node = Node(True, np.unique(data)[0])  # set node as leaf, set class as survived or dead
            self.node_count += 1
            self.leaves.append(node)
            return node
        elif len(split_candidates) == 0:
            # can't split anymore or can't go further because of max level
            # need to take majority for max depth
            vals, counts = np.unique(data, return_counts=True)
            node = Node(True, vals[np.argmax(counts)])
            self.node_count += 1
            self.leaves.append(node)
            return node
        elif self.model_type == "depth" and self.max_depth == level:
            vals, counts = np.unique(data, return_counts=True)
            node = Node(True, vals[np.argmax(counts)])
            self.node_count += 1
            self.leaves.append(node)
            return node
        else:
            min_splits = []
            for candidate in split_candidates:
                if candidate[1] in self.continuous_attributes:
                    # candidate[1] is col index, data is the subset, [candidate[0]] is the value for infgain
                    # for this implementation we are only passing in one value into min_split
                    min_splits.append(self.min_split_continuous(candidate[1], data, np.array([candidate[0]])))
                else:
                    min_splits.append(self.min_split_discrete(candidate[1], data, np.array([candidate[0]])))

            # min_split has all the candidates and information gains [(.566324245235, (0, 1)), (.566324245235, (0, 1))]
            min_split = max(min_splits, key=lambda t: t[0])

            if min_split[0] == 0:
                vals, counts = np.unique(data, return_counts=True)
                node = Node(True, vals[np.argmax(counts)])
                self.node_count += 1
                self.leaves.append(node)
                return node

            for j, candidate in enumerate(split_candidates): # j is index, candidate is value
                # if the value and the column match we found what we want to delete
                if min_split[2] == candidate[1] and min_split[1] == candidate[0]:
                    del split_candidates[j] # delete it
                    break

            node = Node()
            self.node_count += 1
            node.feature = min_split[2]
            node.value = min_split[1]
            node.parent = None

            node.continuous = True if node.feature in self.continuous_attributes else False
            self.non_leaves.append(node)
            node.leaf = False

            vals, counts = np.unique(data, return_counts=True)
            node.cls = vals[np.argmax(counts)]

            # add else statements to make majority leaf

            if not node.continuous:
                sub_yes = data[data[:, node.feature] == node.value]
                if sub_yes.shape[0] > 0:
                    node.yes = self.build_tree(sub_yes, split_candidates, level + 1)
                else:
                    vals, counts = np.unique(data, return_counts=True)
                    node.yes = Node(True, vals[np.argmax(counts)])
                    self.node_count += 1
                    self.leaves.append(node.yes)

                sub_no = data[data[:, node.feature] != node.value]
                if sub_no.shape[0] > 0:
                    node.no = self.build_tree(sub_no, split_candidates, level + 1)
                else:
                    vals, counts = np.unique(data, return_counts=True)
                    node.no = Node(True, vals[np.argmax(counts)])
                    self.node_count += 1
                    self.leaves.append(node.no)
            else:
                sub_yes = data[data[:, node.feature] > node.value]
                if sub_yes.shape[0] > 0:
                    node.yes = self.build_tree(sub_yes, split_candidates, level + 1)
                else:
                    vals, counts = np.unique(data, return_counts=True)
                    node.yes = Node(True, vals[np.argmax(counts)])
                    self.node_count += 1
                    self.leaves.append(node.yes)

                sub_no = data[data[:, node.feature] <= node.value]
                if sub_no.shape[0] > 0:
                    node.no = self.build_tree(sub_no, split_candidates, level + 1)
                else:
                    vals, counts = np.unique(data, return_counts=True)
                    node.no = Node(True, vals[np.argmax(counts)])
                    self.node_count += 1
                    self.leaves.append(node.no)

            if node.yes is not None:
                node.yes.parent = node
            if node.no is not None:
                node.no.parent = node

            if level == 0:
                self.root = node

            # print(level)

            return node

    def prune(self, node, validation_set):

        before_cut = self.calculate_accuracy(validation_set[:, 0:7], validation_set[:, [7]])
        temp_yes = node.yes
        temp_no = node.no

        node.yes = None
        node.no = None
        node.leaf = True

        after_cut = self.calculate_accuracy(validation_set[:, 0:7], validation_set[:, [7]])

        self.prune_recursion(node, validation_set, before_cut, temp_yes, temp_no, after_cut)

    def prune_recursion(self, node, validation_set, before_cut, temp_yes, temp_no, after_cut):
        if after_cut <= before_cut:
            node.yes = temp_yes
            node.no = temp_no
            node.leaf = False
            if node.yes is not None and not node.yes.leaf:
                self.prune(node.yes, validation_set)
            if node.no is not None and not node.no.leaf:
                self.prune(node.no, validation_set)

    def plot_vanilla_first(self):
        mat.scatter(self.vanilla_training_set_percentage, self.vanilla_training_set_accuracy)
        mat.scatter(self.vanilla_training_set_percentage, self.vanilla_test_set_accuracy)
        mat.title('Training Size vs. Accuracy')
        mat.ylabel('Accuracy')
        mat.xlabel('Training Size Percentage')
        mat.show()

    def plot_vanilla_second(self):
        # print(self.node_count)
        mat.scatter(self.vanilla_training_set_percentage, self.nodes)
        mat.title('Nodes vs. Training Set Percentage')
        mat.ylabel('Number of Nodes')
        mat.xlabel('Training Size Percentage')
        mat.show()

    def plot_depth_first(self):
        mat.scatter(self.depth_training_set_percentage, self.depth_training_set_accuracy_max)
        mat.scatter(self.depth_training_set_percentage, self.depth_test_set_accuracy_max)
        mat.title('Training Size vs. Accuracy')
        mat.ylabel('Accuracy')
        mat.xlabel('Training Size Percentage')
        mat.show()

    def plot_depth_second(self):
        # print(self.node_count)
        mat.ylim(0, 350)
        mat.scatter(self.depth_training_set_percentage, [287, 287, 287, 287, 287])
        mat.title('Nodes vs. Training Set Percentage')
        mat.ylabel('Number of Nodes')
        mat.xlabel('Training Size Percentage')
        mat.show()

    def plot_depth_third(self):
        # print(self.node_count)
        mat.scatter(self.depth_training_set_percentage, self.depth_depth_max)
        mat.title('Depth vs. Training Set Percentage')
        mat.ylim(0, 30)
        mat.ylabel('Depth')
        mat.xlabel('Training Size Percentage')
        mat.show()

    def plot_prune_first(self):
        mat.scatter(self.prune_training_set_percentage, self.prune_training_set_accuracy)
        mat.scatter(self.prune_training_set_percentage, self.prune_test_set_accuracy)
        mat.title('Training Size vs. Accuracy')
        mat.ylabel('Accuracy')
        mat.xlabel('Training Size Percentage')
        mat.show()

    def plot_prune_second(self):
        # print(self.node_count)
        mat.ylim(0, 350)
        mat.scatter(self.prune_training_set_percentage, self.nodes)
        mat.title('Nodes vs. Training Set Percentage')
        mat.ylabel('Number of Nodes')
        mat.xlabel('Training Size Percentage')
        mat.show()

    def min_split_continuous(self, col_index, matrix, values):

        dead = matrix[matrix[:, 7] == 0].shape[0]  # matrix of values which contain people who had not survived
        alive = matrix[matrix[:, 7] == 1].shape[0]  # matrix of values which contain people who had survived

        gains = []

        for value in values:
            subset_yes = matrix[matrix[:, col_index] > value]  # subset of values which are greater than split value

            dead_sub_yes = subset_yes[subset_yes[:, 7] == 0].shape[0]  # yes and dead
            alive_sub_yes = subset_yes[subset_yes[:, 7] == 1].shape[0]  # yes and survived

            # subset of values which are less than and equal to split value
            subset_no = matrix[matrix[:, col_index] <= value]

            dead_sub_no = subset_no[subset_no[:, 7] == 0].shape[0]  # no and dead
            alive_sub_no = subset_no[subset_no[:, 7] == 1].shape[0]  # no and survived

            # array of information gains of each value in each column
            gains.append(infor_gain([dead, alive], [[dead_sub_yes, alive_sub_yes], [dead_sub_no, alive_sub_no]]))

        value_max_gain = values[np.argmax(np.array(gains))]  # max information gain value
        #print(value_max_gain)
        #print(np.max(np.array(gains)))

        return np.max(np.array(gains)), value_max_gain, col_index  # return value of max gain, info gain of each, column

    def min_split_discrete(self, col_index, matrix, values):

        dead = matrix[matrix[:, 7] == 0].shape[0]
        alive = matrix[matrix[:, 7] == 1].shape[0]

        gains = []

        for value in values:
            subset_yes = matrix[matrix[:,col_index] == value]

            dead_sub_yes = subset_yes[subset_yes[:, 7] == 0].shape[0]
            alive_sub_yes = subset_yes[subset_yes[:, 7] == 1].shape[0]

            subset_no = matrix[matrix[:, col_index] != value]

            dead_sub_no = subset_no[subset_no[:, 7] == 0].shape[0]
            alive_sub_no = subset_no[subset_no[:, 7] == 1].shape[0]

            gains.append(infor_gain([dead, alive], [[dead_sub_yes, alive_sub_yes], [dead_sub_no, alive_sub_no]]))

        value_max_gain = values[np.argmax(np.array(gains))]

        return np.max(np.array(gains)), value_max_gain, col_index

    def print_tree(self, node=None, level=0):
        if node is None:
            node = self.root

        for i in range(level):
            sys.stdout.write('\t')
            sys.stdout.write('|')

        if node.leaf:
            print(node)
        else:
            print(node)
            self.print_tree(node.yes, level + 1) if node.yes is not None else None
            self.print_tree(node.no, level + 1) if node.no is not None else None

    # One sample
    def calculate_accuracy(self, matrix, labels):
        num_correct = 0
        for i, sample in enumerate(matrix):
            val = self.evaluate_tree(sample)
            if val == labels[i][0]:
                num_correct += 1

        return float(num_correct)/matrix.shape[0]

    def evaluate_tree(self, sample, node=None):
        if node is None:
            node = self.root

        if node.leaf is True:
            return node.cls
        else:
            if node.continuous is False:
                if sample[node.feature] == node.value:
                    if node.yes is not None:
                        return self.evaluate_tree(sample, node.yes)
                    else:
                        return None
                else:
                    if node.no is not None:
                        return self.evaluate_tree(sample, node.no)
                    else:
                        return None
            else:
                if sample[node.feature] > node.value:
                    if node.yes is not None:
                        return self.evaluate_tree(sample, node.yes)
                    else:
                        return None
                else:
                    if node.no is not None:
                        return self.evaluate_tree(sample, node.no)
                    else:
                        return None



def ID3():
    pass


if __name__ == "__main__":

    train_folder = sys.argv[1]
    test_folder = sys.argv[2]

    train_file = train_folder + '/titanic-train.data' if train_folder[-1] != '/' else train_folder + 'titanic-train.data'
    train_label = train_folder + '/titanic-train.label'
    test_file = test_folder + '/titanic-test.data'
    test_label = test_folder + '/titanic-test.label'
    model = sys.argv[3]
    train_set_size = int(sys.argv[4])

    training_data = pd.read_csv(train_file, delimiter=',', engine='python')
    testing_data = pd.read_csv(test_file, delimiter=',', engine='python')

    training_data_matrix = training_data.values
    testing_data_matrix = testing_data.values

    training_label = pd.read_csv(train_label, delimiter=' ', index_col=None, header=None, engine='python')
    testing_label = pd.read_csv(test_label, delimiter=',', index_col=None, header=None, engine='python')

    columns = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked' 'relatives', 'IsAlone']

    if model == "vanilla":
        # print(training_data, training_label, training_data_matrix, testing_data, testing_label, model, train_set_size)

        dt = Tree(model, train_set_size)

        dmatrix = np.concatenate((training_data_matrix, training_label.values), axis=1)

        end_index = int(float(dmatrix.shape[0]) * train_set_size / 100)
        train_matrix = dmatrix[0:end_index, :]

        dt.build_tree(train_matrix)

        print("Train set accuracy: {0:.4f}".format(dt.calculate_accuracy(train_matrix[:,0:7], train_matrix[:,[7]])))
        print("Test set accuracy: {0:.4f}".format(dt.calculate_accuracy(testing_data_matrix, testing_label.values)))

        # dt.plot_vanilla_second()

    if model == "depth":
        validation_set = sys.argv[5]
        max_depth = sys.argv[6]

        dt = Tree(model, train_set_size, int(max_depth))

        dmatrix = np.concatenate((training_data_matrix, training_label.values), axis=1)

        end_index_train = int(float(dmatrix.shape[0]) * train_set_size / 100)
        train_matrix = dmatrix[0:end_index_train, :]

        start_index_validation = int(float(dmatrix.shape[0]) * (1 - (float(train_set_size) / 100)))
        validation_matrix = dmatrix[start_index_validation:, :]

        dt.build_tree(dmatrix)

        print("Train set accuracy: {0:.4f}".format(dt.calculate_accuracy(train_matrix[:,0:7], train_matrix[:,[7]])))
        print("Validation set accuracy: {0:.4f}".format(dt.calculate_accuracy(validation_matrix[:,0:7], validation_matrix[:,[7]])))
        print("Test set accuracy: {0:.4f}".format(dt.calculate_accuracy(testing_data_matrix, testing_label.values)))

        # dt.plot_depth_third()

    if model == "min_split":
        validation_set = sys.argv[5]
        sample_split_size_arg = sys.argv[6]

        dt = Tree(model, train_set_size, sample_split_size=int(sample_split_size_arg))

        dmatrix = np.concatenate((training_data_matrix, training_label.values), axis=1)

        end_index_train = int(float(dmatrix.shape[0]) * train_set_size / 100)
        train_matrix = dmatrix[0:end_index_train, :]

        start_index_validation = int(float(dmatrix.shape[0]) * (1 - (float(train_set_size) / 100)))
        validation_matrix = dmatrix[start_index_validation:, :]

        dt.build_tree(dmatrix)

        print("Train set accuracy: {0:.4f}".format(dt.calculate_accuracy(train_matrix[:, 0:7], train_matrix[:, [7]])))
        print("Validation set accuracy: {0:.4f}".format(
            dt.calculate_accuracy(validation_matrix[:, 0:7], validation_matrix[:, [7]])))
        print("Test set accuracy: {0:.4f}".format(dt.calculate_accuracy(testing_data_matrix, testing_label.values)))

    if model == "prune":
        training_set = sys.argv[4]
        validation_set = sys.argv[5]

        dt = Tree(model, train_set_size)

        dmatrix = np.concatenate((training_data_matrix, training_label.values), axis=1)

        end_index_train = int(float(dmatrix.shape[0]) * train_set_size / 100)
        train_matrix = dmatrix[0:end_index_train, :]

        start_index_validation = int(float(dmatrix.shape[0]) * (1 - (float(train_set_size) / 100)))
        validation_matrix = dmatrix[start_index_validation:, :]

        dt.build_tree(dmatrix)

        # print("Train set accuracy: {0:.4f}".format(dt.calculate_accuracy(train_matrix[:, 0:7], train_matrix[:, [7]])))
        # print("Validation set accuracy: {0:.4f}".format(dt.calculate_accuracy(validation_matrix[:, 0:7], validation_matrix[:, [7]])))
        # print("Test set accuracy: {0:.4f}".format(dt.calculate_accuracy(testing_data_matrix, testing_label.values)))
        # print()

        dt.prune(dt.root, validation_matrix)

        print("Train set accuracy: {0:.4f}".format(dt.calculate_accuracy(train_matrix[:, 0:7], train_matrix[:, [7]])))
        #print("Validation set accuracy: {0:.4f}".format(dt.calculate_accuracy(validation_matrix[:, 0:7], validation_matrix[:, [7]])))
        print("Test set accuracy: {0:.4f}".format(dt.calculate_accuracy(testing_data_matrix, testing_label.values)))

        # dt.plot_prune_second()

    # build decision tree

    # predict on testing set & evaluate the testing accuracy