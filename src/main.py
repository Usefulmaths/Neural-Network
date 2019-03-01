from utilities import load_data
from neural_network import NeuralNetwork
from activations import Softmax, Sigmoid, Relu
from loss_functions import LossFunctions
import numpy as np
import matplotlib.pyplot as plt

# Load in the MNIST data using the utility function load_data.
X_train, X_test, y_train, y_test = load_data()

# Normalise the pixel values of the X data.
X_train, X_test = np.divide(X_train, 255.0), np.divide(X_test, 255.0)

# Define the activation functions on the hidden and final layers,
# and the cost function.
activation_functions = [Sigmoid, Relu, Softmax]
loss_function = LossFunctions.cross_entropy

# Define the number of layers and the number of neurons in each layer of
# the network.
layers = [784, 100, 50, 10]

# Specify properties of the network and training.
epochs = 50
learning_rate = 0.2
batch_size = 256
initialiser = 'xavier'

# Instantiate a NeuralNetwork object with specified properties.
nn = NeuralNetwork(layers=layers,
                   activation_functions=activation_functions,
                   initialiser=initialiser)

# Train the neural network and store the accuracies and losses.
train_losses, test_losses, train_accuracies, test_accuracies = \
    nn.train(
        X_data=X_train,
        y_data=y_train,
        loss_function=loss_function,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        X_validation=X_test,
        y_validation=y_test
    )

# Plot the accuracies and losses
plt.plot(train_losses, label="Self implemented train loss")
plt.plot(test_losses, label="Self implemented test loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Cross-Entropy Loss")
plt.title("The cross entropy loss vs iterations.")
plt.legend()
plt.show()

plt.plot(train_accuracies, label="Train accuracy")
plt.plot(test_accuracies, label="Test accuracy")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy vs iterations")
plt.legend(loc="lower right")
plt.show()
