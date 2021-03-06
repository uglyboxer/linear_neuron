# Author: Cole Howard
#
# network.py is a basic implementation of a one layer linear neural network, to
# examine an implementation of backpropagation.  It is based on the basic model
# of the Perceptron.  Information on that can be found at:
# https://en.wikipedia.org/wiki/Perceptron
#  The intent of this specific project is to alter the Perceptron's
# decision function to a logistic function and add a "backpropagation" step
# at the end of each vector's pass through the neuron.
#
# There are several methods included that are currently passed, as they are
# plans to make it extensible as possible as a tool for exploring neural nets
# under more general circumstances.
#
# Usage:
#   It is currently set up to run a training set of input (along with the
#   associated answers) and a set of similar, but distinct, input (without)
#   the answers, and have the machine guess an answer to each of those cases
#   based on information it gathered during the training run.
#
#   To execute as is, from the command line, while in the linear_neuron/source/
#   directory, input:
#
#       $ python3 network.py
#
#   This will pull the learning and test data from scikit-learn run both and
#   return a success count, total count of the unseen test data, and the
#   success rate that equals.
#
#   First output and success ratio will be based on the first set of testing
#   vectors.  The second set will represent the same for the validation
#   set.  The visualization (see below) that pops up, just close that window
#   for the script to finish running.
#
# Alternate data sets:
#   Alternate training and testing data sets can be swapped out in the first
#   section of main() below.  See those notes for specifics.
#
# Visualization:
#  Pyplot is included to provide a visual representation of a memember of the
#  dataset.

from random import choice
from time import sleep

from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets, utils

from feedback import feedback
from neuron import Neuron


class Network:
    """ A Network instance will create layers of neurons for the implementa-
    tion of neural network.

    Parameters
    ----------
    images : list
        Corresponding images of the dataset
    neuron_targets : list
        The possible final output values
    vector_size : int
        Size of the individual input vectors
    train_set : list
        Set of vectors for the learning portion
    train_answers : list
        Correct answers that correspond to the train_set
    test_set : list
        Set of vectors, discrete from the train_set to have the machine
        guess against
    test_answers : list
        Correct answers for the test_set, to compare the machine's
        guesses against
    validation_set : list
        A validation set to compare answers in a second run
    validation_answers : list
        Answer for the above

    Attributes
    ----------
    neurons : Class Neuron
        Instances of the Neuron class, one for each of possible correct
        answers
    """

    def __init__(self, images, neuron_targets, vector_size, train_set,
                 train_answers, test_set, test_answers, validation_set,
                 validation_answers):

        self.images = images
        self.neuron_count = neuron_targets  
        self.vector_size = vector_size
        self.train_set = train_set
        self.train_answers = train_answers
        self.test_set = test_set
        self.test_answers = test_answers
        self.validation_set = validation_set
        self.validation_answers = validation_answers
        self.neurons = [Neuron(self.vector_size, x, len(self.train_set),
                        self.train_answers) for x in self.neuron_count]

    def gradient_descent(self, vector, vector_index):
        """ Calculates the gradient_descent

        Parameters
        ----------
        vector : list
            A single input, comprised of floats
        vector_index : int

        Attributes
        ----------
        learning_rate : float
            Determines how much of the error is applied to the weights
            in each iteration

        Returns
        -------
        float
            Represents the error to be used to update the weights of
            the neurons.  It should approximate a gradient descent in
            topology of the outputs
        """

        learning_rate = .05
        temp_list = [(self.neurons[x]._sigmoid(self.neurons[x]._dot_product(
                     vector)) - self.neurons[x].expected[vector_index]) *
                     self.neurons[x]._sigmoid(
                     self.neurons[x]._dot_product(vector)) * (1 -
                     self.neurons[x]._sigmoid(self.neurons[x]._dot_product(
                      vector))) for x in self.neuron_count]
        gd = -1 * learning_rate * sum(temp_list)
        return gd

    def learn_run(self):
        """ Runs an iteration through the neuron sets and adjust the weights
        appropriately.  It then follows up with a second weight adjusment
        accross all neurons with an estimate of the gradient descent
        function
        """

        for idx, vector in enumerate(self.train_set):
            for neuron in self.neurons:
                neuron.train_pass(vector, idx)
            gd = self.gradient_descent(vector, idx)   # Backpropogate the error
            for neuron in self.neurons:
                neuron.update_weights(gd, vector)

    def run_unseen(self, validation=False):
        """ Makes guesses on the unseen data, and switches over the test
        answers to validation set if the bool is True

        For each vector in the collection, each neuron in turn will either
        fire or not.  If a vector fires, it is collected as a possible
        correct guess.  Not firing is collected as well, in case
        there an no good guesses at all.  The method will choose the
        vector with the highest dot product, from either the fired list
        or the dud list.

        Parameters
        ----------
        validation : bool
            Runs a different set of vectors through the guessing
            process if validation is set to True

        Returns
        -------
        list
            a list of ints (the guesses for each vector)
        """

        if validation:
            self.test_set = self.validation_set
        temp_guess_list = [[] for x in self.test_set]
        temp_dud_guess_list = [[] for x in self.test_set]
        for idy, vector in enumerate(self.test_set):

            for idx, neuron in enumerate(self.neurons):
                nf = neuron.fires(vector)
                if nf[0]:
                    temp_guess_list[idy].append((nf[1], idx))
                    temp_dud_guess_list[idy].append((0, idx))
                else:
                    temp_guess_list[idy].append((0, None))
                    temp_dud_guess_list[idy].append((nf[1], idx))
            temp_guess_list[idy].sort(reverse=True)
            temp_dud_guess_list[idy].sort(reverse=True)
        guess_list = [x[0][1] for x in temp_guess_list]
        dud_guess_list = [x[0][1] for x in temp_dud_guess_list]
        new_guess_list = [x if (x is not None) else dud_guess_list[idx] for
                          idx, x in enumerate(guess_list)]
        return new_guess_list

    def report_results(self, guess_list, validation=False):
        """ Reports results of guesses on unseen set

        Parameters
        ----------
        guess_list : list

        """
        if validation:
            self.test_answers = self.validation_answers
            print("I guess this is a: ", guess_list[1])
            plt.imshow(self.images[1451], cmap="Greys",
                       interpolation='nearest')
            plt.show()
        successes = 0
        for idx, item in enumerate(guess_list):
            if self.test_answers[idx] == item:
                successes += 1
        print("Successes: {}  Out of total: {}".format(successes,
              len(guess_list)))
        print("For a success rate of: ", successes/len(guess_list))


def append_bias(vector):
    """ Takes a list of n entries and appends a 1 for the bias

    Parameters
    ----------
    vector : list

    Attributes
    ----------
    num_of_training_vectors : int
        This is to adjust the size of the training set when all of the data
        is provided as large list.  Breaking the training data into a
        training set, testing set, and a validation set.  Picking this number
        is a balance between speed (lower number) and overfitting the data
        (a higher number)

    Returns
    -------
    list
        The input vector with a one appended to the end of the list, as
        a bias
    """
    temp_vector = [x for x in vector]
    temp_vector.append(1)
    return temp_vector


def visualization(vector, vector_name):
    y = np.reshape(vector, (8,8))
    plt.imshow(y, cmap=cm.jet)#cm.Greys_r)
    plt.suptitle(vector_name)
    plt.axis('off')
    plt.pause(0.0001)
    plt.show()
    # sleep(.2)
    # plt.close()


def main():

    # In the scikit-learn set below, the data is shuffled using utils.resample
    # as the first pass had an artifact in the end of the list that wasn't
    # representative of the rest of the set.

    # Dependent on input set
    temp_digits = datasets.load_digits()
    digits = utils.resample(temp_digits.data, random_state=0)
    temp_answers = utils.resample(temp_digits.target, random_state=0)
    images = utils.resample(temp_digits.images, random_state=0)
    target_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    num_of_training_vectors = 1000
    answers, answers_to_test, validation_answers = temp_answers[:num_of_training_vectors], temp_answers[num_of_training_vectors:num_of_training_vectors+500], temp_answers[num_of_training_vectors+500:]
    training_set, testing_set, validation_set = digits[:num_of_training_vectors], digits[num_of_training_vectors:num_of_training_vectors+500], digits[num_of_training_vectors+500:]

    # For all inputs
    training_vectors = [append_bias(vector) for vector in training_set]
    test_vectors = [append_bias(vector) for vector in testing_set]

    network = Network(images, target_values, len(training_set[0]),
                      training_vectors, answers, test_vectors, answers_to_test,
                      validation_set, validation_answers)
    # [network.learn_run() for x in range(250)]  # Number of Epochs over set
        # network.report_results(network.run_unseen())
        # network.report_results(network.run_unseen(True), True)
    
    # plt.ion()
    # with open("weight_set.txt", 'w') as w:
    #     for elem in network.neurons:
    #         for x in elem.weights:
    #             y = str(x) + ', '
    #             w.write(y)

    with open("weight_set.txt", 'r') as w:
        for line in w:
            weights = line.strip().split(', ')
    

    w_list = [float(elem.replace(',', '')) for elem in weights]
    for x in range(10):
        network.neurons[x].weights = w_list[:64]
        w_list = w_list[64:] 
    print(network.neurons[1].weights)
    x = np.random.normal(128, 1, (8, 8))
    y = x.flatten()
    # y = network.test_set[0][:-1]
    network.test_set = [y]
    visualization(y, 'Gaussian Noise')
    guess = network.run_unseen()
    print(guess)
    # print(y)
    # input("Hit a key to continue")
    target_values.remove(guess[0])
    for i in range(15):
        for i in range(5):
            y = feedback(y, network.neurons[guess[0]].weights, .001)
            for x in target_values:
                y = feedback(y, network.neurons[x].weights, -.0001)
        visualization(y, 'Dreamt a ' + str(guess[0]))
        network.test_set = [y]
        guess = network.run_unseen()
        print(guess)
    y = [(neurons[4].x)%256 for x in weights]
    visualization(y, 'Neuron 4 weight set')

if __name__ == '__main__':
    main()
