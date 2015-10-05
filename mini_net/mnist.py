""" Adapted further from http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py

Then runs it through mini_net's network of neurons.

"""

import os, struct
from array import array as pyarray
import numpy as np
from numpy import append, array, int8, ndarray, uint8, zeros

from network import Network


def load_mnist(dataset="training", digits=np.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()
    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]
    imgs = []
    for i in images:
        imgs.append(i.flatten())

    lbls = []
    for i in labels:
        lbls.append(i.flatten())

    return imgs, lbls


if __name__ == '__main__':
    target_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    epoch = 1
    training_set, answers = load_mnist()
    testing_set, answers_to_test = load_mnist("testing")
    validation_set, validation_answers = load_mnist("testing")

    network = Network(target_values, training_set, answers, epoch, testing_set,
                      answers_to_test, validation_set, validation_answers)
    network.learn_run()
    network.report_results(network.run_unseen())