import numpy as np
from model import Model

# Prepare your data
X = np.random.randn(10, 100)  # 10 features, 100 examples
Y = np.random.randint(0, 2, size=(1, 100))  # binary labels for 100 examples

# Define your model
layer_dims = [10, 5, 1]  # 10-input features, 5-neurons in hidden layer, 1-output neuron
activations = ["relu", "softmax"]  # ReLU activation for hidden layer, Softmax for output layer

model = Model(layer_dims, activations)

# Train the model
model.train(X, Y, epochs=1000, learning_rate=0.01)

# Evaluate the model
predictions = model.predict(X)
accuracy = np.mean(predictions == Y)
print(f"Accuracy: {accuracy * 100}%")
