# Convolutional Neural Network from Scratch

This project is a fundamental and mathematical implementation of a Convolutional Neural Network (CNN) from scratch using NumPy. It encompasses all core components of a standard neural network, including forward and backward propagation, convolutional and pooling layers, fully connected layers, and activation functions. The implementation uses the principles of linear algebra, calculus, and statistics, providing a deep insight into the underpinning mechanics of deep learning models. This project serves as a practical and educational resource for understanding the computational and mathematical complexities involved in a CNN.
## Getting Started

Just run the `main.py` script. It will train a CNN model on a binary classification problem and print the cost after each 100 epochs and the final accuracy of the model.
```shell
python main.py
```
You should see an output similar to this:
```shell
Cost after epoch 0: 10.868201634831896
Cost after epoch 100: 10.868201634831896
Cost after epoch 200: 10.868201634831896
Cost after epoch 300: 10.868201634831896
Cost after epoch 400: 10.868201634831896
Cost after epoch 500: 10.868201634831896
Cost after epoch 600: 10.868201634831896
Cost after epoch 700: 10.868201634831896
Cost after epoch 800: 10.868201634831896
Cost after epoch 900: 10.868201634831896
Training completed.
Accuracy: 41.0%
```

## Structure of the Project

- `activations.py`: Contains implementation of ReLU and Softmax activation functions.
- `layer.py`: Contains implementation of Convolutional layer, Pooling layer, and Fully Connected layer.
- `network.py`: Constructs a neural network using the layers defined in `layer.py`.
- `model.py`: Defines the Model class which uses the Network class for training and prediction.
- `utils.py`: Contains utility functions for image padding and convolution operations.
- `main.py`: The main script that uses all the above components to train and evaluate the CNN.

## Dependencies

- NumPy

## License

This project is open-source and available under the MIT License.
