from layer import ConvLayer, PoolLayer, FCLayer
from activations import relu, softmax

class Network:
    """
    This class defines a simple feedforward neural network.
    """

    def __init__(self, layers):
        """
        Initialize the network with a list of Layer objects.

        Args:
            layers (list): List of Layer objects (ConvLayer, PoolLayer, FCLayer etc.).
        """
        self.layers = layers

    def forward(self, X):
        """
        Perform a forward pass through the network.

        Args:
            X (numpy.ndarray): Input data.

        Returns:
            A (numpy.ndarray): Output from the last layer.
            caches (list): List of cache data from each layer, needed for backpropagation.
        """
        caches = []
        A = X

        for layer in self.layers:
            A, cache = layer.forward(A)
            caches.append(cache)

        return A, caches

    def backward(self, dA, caches):
        """
        Perform a backward pass through the network.

        Args:
            dA (numpy.ndarray): Upstream gradients.
            caches (list): List of cache data from each layer.

        Returns:
            grads (dict): A dictionary with the gradients for each layer.
        """
        grads = {}
        L = len(caches)

        for l in reversed(range(L)):
            current_cache = caches[l]
            dA, dW, db = self.layers[l].backward(dA, current_cache)
            grads["dW" + str(l+1)] = dW
            grads["db" + str(l+1)] = db

        return grads
