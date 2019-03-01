import numpy as np
from activations import Sigmoid, Relu, Softmax


class NeuralNetwork(object):
    """
    A class that represents a neural network. Methods consist of
    initialising parameters, performing forward passes, performing
    backpropagation, and training the network. Generalised for any
    number of layers and neurons as well as any order of activation
    function in the hidden layer (activations defined in activations.py).
    Softmax layer must be last.
    """

    def __init__(self, layers, activation_functions=[Sigmoid, Relu, Softmax],
                 initialiser='xavier'):

        self.layers = layers

        self.initialiser = initialiser
        self.weights, self.biases = self.initialise_network_parameters()
        self.activation_functions = activation_functions

    def initialise_network_parameters(self):
        """
        Generates random initial parameters for the network.
        Currently able to perform two types of initialisation,
        standard_normal and xavier.
        Returns:
                weights -- an array consisting of the weight matrices
                of each layer
                biases -- an array consisting of the bias vectors
                of each layer
        """
        weights = []
        biases = []
        for index in range(len(self.layers) - 1):
            current_layer = self.layers[index]
            next_layer = self.layers[index + 1]

            if(self.initialiser == 'standard_normal'):
                weights.append(np.random.randn(current_layer, next_layer))

            elif(self.initialiser == 'xavier'):
                mu = 0
                sigma = np.sqrt(2. / (current_layer + next_layer))
                initial_weights = np.random.normal(mu, sigma,
                                                   (current_layer, next_layer)
                                                   )
                weights.append(initial_weights)

            biases.append(np.zeros((current_layer, next_layer)))

        return weights, biases

    def forward_pass(self, X_data):
        """
        Performs a forward pass of the neural network
        Arguments:
                X_data -- the data to be passed through the network.
        Returns:
                current_activation -- the predicted activation of the
                neural network.
                zs -- an array consisting of all Z values in
                each layer.
                activations -- an array consisting of all activations in
                each layer.
        """
        current_activation = X_data

        zs = []
        activations = [X_data]

        for index in range(len(self.layers) - 1):
            weight = self.weights[index]
            bias = self.biases[index]
            activation = self.activation_functions[index].function

            z = np.dot(current_activation, weight) + bias[0]
            a = activation(z)

            zs.append(z)
            activations.append(a)

            current_activation = a

        return current_activation, zs, activations

    def calculate_derivatives(self, activations, errors):
        """
        Calculates the derivatives of the loss function in respect to
         all the parameters in the network.
        Arguments:
                activations -- an array consisting of all activations
                in each layer.
                errors -- an array consistng of all the error signals
                in each layer.
        Returns:
                d_weights -- the derivative of the cost function in respect
                to the weights.
                d_biases -- the derivative of the cost function in respect
                to the biases.
        """
        d_weights = []
        d_biases = []

        for layer in range(len(self.layers) - 2, -1, -1):
            d_weights.insert(0, np.dot(activations[layer].T, errors[
                layer]) / activations[layer].shape[0])
            d_biases.insert(0, np.mean(errors[layer], axis=0))

        return d_weights, d_biases

    def backpropagation(self, X_data, y_data):
        """
        Performs backpropagation on the neural network to calculate
        the derivatives of the loss function in respect to all the
        parameters in the network given some X and y data.
        Arguments:
                X_data -- X data to be learned
                y_data -- corresponding y data (labels).
        Returns:
                d_weights -- the derivative of the cost function
                in respect to the weights.
                d_biases -- the derivative of the cost function
                in respect to the biases.
        """
        current_activation, zs, activations = self.forward_pass(X_data)

        final_layer_error = (current_activation - y_data)

        errors = [final_layer_error]

        for index in range(len(self.layers) - 2, 0, -1):
            error = np.dot(errors[0], self.weights[
                index].T) * \
                self.activation_functions[index - 1].derivative(zs[index - 1])
            errors.insert(0, error)

        d_weights, d_biases = self.calculate_derivatives(activations, errors)

        return d_weights, d_biases

    def evaluate(self, X_data, y_data):
        """
        Calculates the accuracy of neural networks prediction on
        some data of choice.
        Arguments:
                X_data -- X data to perform predictions on.
                y_data -- y data that corresponds to the true labels.
        Returns:
                accuracy -- the accuracy of the neural networks prediction.

        """
        predictions, _, _ = self.forward_pass(X_data)
        number_of_data = X_data.shape[0]

        correct = 0

        for prediction, actual in zip(predictions, y_data):
            pred_index = np.argmax(prediction)
            actual_index = np.argmax(actual)
            if(actual_index == pred_index):
                correct += 1

        accuracy = float(correct) / number_of_data
        return accuracy

    def train(self, X_data, y_data, loss_function, epochs, learning_rate,
              batch_size, X_validation, y_validation):
        """
        Performs stochastic gradient descent to train the neural network.
        Arguments:
                X_data -- X data to be learned
                y_data -- corresponding y data (labels).
                epochs -- number of epochs to train for.
                learning_rate -- the step size to be taken in SGD.
                batch_size -- the batch size to be fed into backpropagation.
                X_validation -- X data from validation set to test the
                performance of the network.
                y_validation -- y data from validation set to test the
                performance of the network.
        Returns:
                training_losses -- an array consisting of each loss on the
                training set at every stage of SGD.
                test_losses -- an array consisting of each loss on the
                 test set at every stage of SGD.
                train_accuracies -- an array consisting of each accuracy on the
                 training set at every stage of SGD.
                test_losses -- an array consisting of each accuracy on the
                 test set at every stage of SGD.
        """
        train_losses = []
        test_losses = []
        test_accuracies = []
        train_accuracies = []

        number_of_data = X_data.shape[0]

        training_data = zip(X_data, y_data)

        for epoch in range(epochs):
            np.random.shuffle(training_data)

            mini_batches = [training_data[k:k + batch_size]
                            for k in range(0, number_of_data, batch_size)]

            for mini_batch in mini_batches:
                X_batch = np.array(map(lambda x: x[0], mini_batch))
                y_batch = np.array(map(lambda x: x[1], mini_batch))
                d_weights, d_biases = self.backpropagation(X_batch, y_batch)

                for weight_index in range(len(self.weights)):
                    self.weights[weight_index] = self.weights[
                        weight_index] - learning_rate * d_weights[weight_index]
                    self.biases[weight_index] = self.biases[
                        weight_index] - learning_rate * d_biases[weight_index]

            test_prediction, _, _ = self.forward_pass(X_validation)
            test_loss = loss_function(y_validation, test_prediction)
            test_accuracy = self.evaluate(X_validation, y_validation)

            train_prediction, _, _ = self.forward_pass(X_data)
            train_loss = loss_function(y_data, train_prediction)
            train_accuracy = self.evaluate(X_data, y_data)

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            print("Epoch: %5d, Training Set Accuracy: %8.4f, Training Set Loss: %8.4f, \
                Test Set Accuracy: %8.4f, Test Set Loss: %8.4f" %
                  (epoch, train_accuracy, train_loss,
                   test_accuracy, test_loss))

        return train_losses, test_losses, train_accuracies, test_accuracies
