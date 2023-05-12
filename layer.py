from abc import ABC, abstractmethod
import numpy as np
from utils import zero_pad, conv_single_step

class Layer(ABC):
    @abstractmethod
    def forward(self, input):
        pass

    @abstractmethod
    def backward(self, output_grad):
        pass

class ConvLayer(Layer):
    """
    This class defines a convolutional layer.
    It inherits from the abstract base class "Layer".
    """
    def __init__(self, filters, filter_size, stride, pad, input_dim):
        self.filters = filters
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.W = np.random.randn(filter_size, filter_size, input_dim, filters)
        self.b = np.zeros((1, 1, 1, filters))

    def forward(self, A_prev):
        """
        Implements the forward propagation for a convolution function

        Arguments:
        A_prev -- output activations of the previous layer, 
        numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)

        Returns:
        Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache of values needed for the conv_backward() function
        """

        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        n_H = int((n_H_prev - self.filter_size + 2 * self.pad) / self.stride) + 1
        n_W = int((n_W_prev - self.filter_size + 2 * self.pad) / self.stride) + 1

        Z = np.zeros((m, n_H, n_W, self.filters))
        A_prev_pad = zero_pad(A_prev, self.pad)

        for i in range(m):
            a_prev_pad = A_prev_pad[i]
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(self.filters):
                        vert_start = h * self.stride
                        vert_end = vert_start + self.filter_size
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + self.filter_size

                        a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                        Z[i, h, w, c] = conv_single_step(a_slice_prev, self.W[...,c], self.b[...,c])

        cache = (A_prev, self.W, self.b, self.stride, self.pad)

        return Z, cache
    
    def backward(self, dZ, cache):
        """
        Implement the backward propagation for a convolution function

        Arguments:
        dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache of values needed for the conv_backward(), output of conv_forward()

        Returns:
        dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
                   numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        dW -- gradient of the cost with respect to the weights of the conv layer (W)
              numpy array of shape (f, f, n_C_prev, n_C)
        db -- gradient of the cost with respect to the biases of the conv layer (b)
              numpy array of shape (1, 1, 1, n_C)
        """
        (A_prev, W, b, stride, pad) = cache
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        (f, f, n_C_prev, n_C) = W.shape
        (m, n_H, n_W, n_C) = dZ.shape

        dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))                           
        dW = np.zeros((f, f, n_C_prev, n_C))
        db = np.zeros((1, 1, 1, n_C))

        A_prev_pad = zero_pad(A_prev, pad)
        dA_prev_pad = zero_pad(dA_prev, pad)

        for i in range(m):                     
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]

            for h in range(n_H):                  
                for w in range(n_W):               
                    for c in range(n_C):           

                        vert_start = h
                        vert_end = vert_start + f
                        horiz_start = w
                        horiz_end = horiz_start + f

                        a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                        dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                        db[:,:,:,c] += dZ[i, h, w, c]
                        
            if pad != 0:
                dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
            else:
                dA_prev[i, :, :, :] = da_prev_pad

        return dA_prev, dW, db

class PoolLayer(Layer):
    """
    This class defines a max pooling layer.
    It inherits from the abstract base class "Layer".
    """
    def __init__(self, filter_size, stride):
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, A_prev):
        """
        Implements the forward propagation for a pooling function

        Arguments:
        A_prev -- Output activations of the previous layer, 
        numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)

        Returns:
        A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
        """

        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        n_H = int(1 + (n_H_prev - self.filter_size) / self.stride)
        n_W = int(1 + (n_W_prev - self.filter_size) / self.stride)
        n_C = n_C_prev

        A = np.zeros((m, n_H, n_W, n_C))

        for i in range(m):
            for h in range(n_H):
                for w in range(n_W):
                    for c in range (n_C):

                        vert_start = h * self.stride
                        vert_end = vert_start + self.filter_size
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + self.filter_size

                        a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                        A[i, h, w, c] = np.max(a_prev_slice)

        cache = (A_prev, self.filter_size, self.stride)

        return A, cache

    def backward(self, dA, cache):
        """
        Implements the backward pass of the max pooling layer

        Arguments:
        dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
        cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 

        Returns:
        dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
        """

        (A_prev, filter_size, stride) = cache
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        m, n_H, n_W, n_C = dA.shape

        dA_prev = np.zeros(A_prev.shape)

        for i in range(m):
            a_prev = A_prev[i]
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):

                        vert_start = h
                        vert_end = vert_start + filter_size
                        horiz_start = w
                        horiz_end = horiz_start + filter_size

                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]

                        mask = a_prev_slice == np.max(a_prev_slice)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i, h, w, c])

        return dA_prev
    
class FCLayer(Layer):
    """
    This class defines a fully connected layer.
    It inherits from the abstract base class "Layer".
    """
    def __init__(self, input_shape, output_shape, activation_func):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.activation_func = activation_func
        self.W = np.random.randn(output_shape, input_shape) * 0.01
        self.b = np.zeros((output_shape, 1))

    def forward(self, A_prev):
        """
        Implements the forward propagation for a fully connected layer

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)

        Returns:
        A -- the output of the activation function, also called the post-activation value 
        cache -- a python dictionary containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
        """

        Z = np.dot(self.W, A_prev) + self.b
        A = self.activation_func(Z)

        cache = (A_prev, self.W, self.b, Z)

        return A, cache

    def backward(self, dA, cache):
        """
        Implement the backward propagation for a single layer.

        Arguments:
        dA -- post-activation gradient for current layer l 
        cache -- tuple of values (A_prev, W, b, Z) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """

        A_prev, W, b, Z = cache
        m = A_prev.shape[1]

        dZ = dA * self.activation_func(Z, derivative=True)
        dW = 1./m * np.dot(dZ, A_prev.T)
        db = 1./m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def update_parameters(self, dW, db, learning_rate):
        """
        Update parameters of the layer by gradient descent

        Arguments:
        dW -- gradient of the cost with respect to the weights of the layer (W)
        db -- gradient of the cost with respect to the biases of the layer (b)
        learning_rate -- learning rate of the gradient descent update rule
        """

        self.W = self.W - learning_rate * dW
        self.b = self.b - learning_rate * db