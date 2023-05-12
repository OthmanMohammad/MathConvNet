import numpy as np

def relu(Z, derivative=False):
    """
    Implements the ReLU function.
    
    Arguments:
    Z -- numpy array of any shape
    derivative -- whether to return the derivative of the function; 
                  if False, return the function value

    Returns:
    result -- the result of applying the ReLU function (or its derivative) to Z
    """
    if derivative:
        return np.where(Z <= 0, 0, 1)
    else:
        return np.maximum(0, Z)

def softmax(Z, derivative=False):
    """
    Implements the softmax function.

    Arguments:
    Z -- numpy array of any shape
    derivative -- whether to return the derivative of the function; 
                  if False, return the function value

    Returns:
    result -- the result of applying the softmax function (or its derivative) to Z
    """
    shiftZ = Z - np.max(Z)
    exps = np.exp(shiftZ)
    softmax = exps / np.sum(exps, axis=0)
    
    if derivative:
        return softmax * (1 - softmax)
    else:
        return softmax
