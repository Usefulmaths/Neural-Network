import numpy as np


class LossFunctions(object):
    """
    A class where different loss functions can be called
    as static methods.
    """

    @staticmethod
    def cross_entropy(actual, prediction):
        """
        Arguments:
                  actual -- the data points labels
                  prediction - the networks prediction of the data points
        Returns:
                 the cross entropy between the labelled data points and
                 the networks predictions.
        """
        return -np.mean(np.sum(np.multiply(actual, np.log(prediction)),
                               axis=1))
