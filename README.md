# Micrograd
This project is a minimal, educational implementation of an automatic differentiation (Autograd) engine, inspired by libraries like PyTorch's torch.Tensor and Andrej Karpathy's Micrograd.

It is built entirely from scratch using Python's fundamental concepts, where the core data type (Value) builds a computational graph to enable efficient backpropagation.

1. The Value Class
The fundamental unit of the library. Every calculation (addition, multiplication, tanh, etc.) involving a Value object creates a new node in the computation graph, linking the output to its inputs via the _prev set.

Data and Gradient: Stores a scalar value (.data) and the calculated derivative (.grad).

Graph Operations: Implements standard arithmetic operators (__add__, __mul__, __sub__, __pow__) and non-linearities (tanh).

Backpropagation: The .backward() method performs a reverse topological sort over the graph, calling the chain-rule-derived _backward() function at each node to distribute gradients.

2. Neural Network Primitives
The library provides basic building blocks for constructing neural networks:

Neuron: A single unit performing a weighted sum and applying the tanh activation function.

layer: A collection of neurons, handling the transition between layers.

MLP (Multi-Layer Perceptron): A sequence of layers, defining the structure of the overall network.

How to Use (Training an MLP)
The complete code demonstrates how to instantiate a network, perform a forward pass, calculate the Mean Squared Error (MSE) loss, and run backpropagation.

1. Forward Pass
The network maps a list of input floats to an output Value object.

# Create a 3-input -> 4-hidden -> 4-hidden -> 1-output network
n = MLP(3, [4, 4, 1]) 

# Get predictions (a list of Value objects)
ypred = [n(x) for x in xs]

2. Loss Calculation and Backpropagation
The gradient must be calculated starting from a single, scalar loss value.

# Reset all existing gradients to zero before starting a new backward pass
for p in n.parameters():
    p.grad = 0.0

# Calculate the MSE loss (creates a single Value object node)
loss = sum([(y_pred - y_target)**2 for y_target, y_pred in zip(ys, ypred)])

# Run backpropagation starting from the final loss node
loss.backward()

3. Optimization
Once loss.backward() is called, every weight's .grad attribute contains  
∂w
∂Loss
​
 . These gradients can then be used to update the weights for training (not implemented in the provided code, but the necessary data is available).

# Example gradient descent step (hypothetical)
learning_rate = 0.01
for p in n.parameters():
    p.data -= learning_rate * p.grad
