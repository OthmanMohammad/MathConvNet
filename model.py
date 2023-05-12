from network import Network
from layer import ConvLayer, PoolLayer, FCLayer
from activations import relu, softmax
import numpy as np

class Model:
    """
    This class builds a model using the Network class and provides functionality 
    for training and prediction.
    """

    def __init__(self, layer_dims, activations):
        """
        Initialize the model with dimensions and activations for each layer.

        Args:
            layer_dims (list): List of layer dimensions. 
            activations (list): List of activation functions for each layer.
        """
        layers = []

        for i in range(len(layer_dims) - 1):
            if activations[i] == "relu":
                layers.append(FCLayer(layer_dims[i], layer_dims[i+1], relu))
            elif activations[i] == "softmax":
                layers.append(FCLayer(layer_dims[i], layer_dims[i+1], softmax))

        self.network = Network(layers)

    def train(self, X, Y, epochs, learning_rate):
        """
        Train the model with the given dataset.

        Args:
            X (numpy.ndarray): Training data.
            Y (numpy.ndarray): Labels for the training data.
            epochs (int): Number of epochs for training.
            learning_rate (float): Learning rate for gradient descent.
        """
        m = X.shape[1]  # number of examples

        for epoch in range(epochs):
            # Forward propagation
            AL, caches = self.network.forward(X)

            # Compute cost
            # The cost function is computed as the cross-entropy loss, which is suitable for binary classification tasks.
            cost = -np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)) / m

            # Backward propagation
            grads = self.network.backward(AL - Y, caches)

            # Update parameters
            for i in reversed(range(len(self.network.layers))):
                self.network.layers[i].W -= learning_rate * grads["dW" + str(i+1)]
                self.network.layers[i].b -= learning_rate * grads["db" + str(i+1)]

            if epoch % 100 == 0:
                print(f"Cost after epoch {epoch}: {cost}")

        print("Training completed.")

    def predict(self, X):
        """
        Predict the output for the given input data.

        Args:
            X (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Predicted output for the input data.
        """
        AL, _ = self.network.forward(X)
        return np.round(AL)  # Round to get binary predictions for binary classification

    def accuracy(self, X, Y):
        """
        Calculate the accuracy of the model for the given data and labels.

        Args:
            X (numpy.ndarray): Input data.
            Y (numpy.ndarray): True labels for the data.

        Returns:
            float: Accuracy of the model's predictions.
        """
        predictions = self.predict(X)
        return (predictions == Y).mean()
