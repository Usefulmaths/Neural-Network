import numpy as np
from abc import abstractmethod


class Activation:
    """
    An abstract class defining the structure of an Activation Function.
    """
    @abstractmethod
    def function(z):
        pass

    @abstractmethod
    def derivative(z):
        pass


class Sigmoid(Activation):
    """
    Sigmoid Activation Function
    """
    @staticmethod
    def function(z):
        return np.divide(1., 1 + np.exp(-z))

    @staticmethod
    def derivative(z):
        return np.multiply(Sigmoid.function(z), 1 - Sigmoid.function(z))


class Relu(Activation):
    """
    Relu Activation Function
    """
    @staticmethod
    def function(z):
        return np.maximum(0, z)

    @staticmethod
    def derivative(z):
        return z > 0


class Softmax(Activation):
    """
    Softmax Activation Function
    """
    @staticmethod
    def function(z):
        shifted = z - np.max(z)
        return np.divide(np.exp(shifted), np.sum(np.exp(shifted),
                                                 axis=1,
                                                 keepdims=True))

    @staticmethod
    def derivative(actual, prediction):
        return None


class Tanh(Activation):
    """
    Tanh Activation Function
    """
    @staticmethod
    def function(z):
        return np.tanh(z)

    @staticmethod
    def derivative(z):
        return np.divide(1., np.cosh(z)**2)
