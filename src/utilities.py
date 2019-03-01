import numpy as np
from mnist import MNIST


def one_hot_encoding(y):
    """
    Calculates the one-hot encoding representation of an integer.
    Arguments:
        y -- the integer to be encoded.
    Returns:
        vec -- a one-hot encoded version of y.
    """
    vec = np.zeros((1, 10))
    vec[0, y] = 1

    return vec


def one_hot_encode_y_data(data):
    """
    Converts all the labels in a data set into a one-hot encoding set.
    Arguments:
        data -- an array of y labels from a dataset.
    Returns:
        y_array -- an array of one-hot encoded converts.
    """
    y_array = []

    for y in data:
        y_array.append(one_hot_encoding(y))

    return y_array


def load_data():
    """
    Loads in the MNIST data and performs one-hot encoding on the y labels.
    Returns:
        X_train -- training X data from MNIST.
        X_test -- test X data from MNIST.
        y_train -- training y labels from MNIST.
        y_test -- test y labels from MNISt.

    """
    mndata = MNIST('../data')
    X_train, y_train = mndata.load_training()
    X_test, y_test = mndata.load_testing()

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    y_train = np.array(one_hot_encode_y_data(y_train)).reshape(-1, 10)
    y_test = np.array(one_hot_encode_y_data(y_test)).reshape(-1, 10)

    return X_train, X_test, y_train, y_test
