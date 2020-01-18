"""
Simple Implementation of forward pass of Multi-layer Perceptron or Feedforward Neural Network

Forward Pass -> XW + b

The way matrix multiplication works is that the column count of the first matrix should be
the same as the row count of the second matrix

Representation of Features in Matrices 

f_1, f_2, f_3, f_4
1,2,3,1
2,1,3,1
5,6,1,3

Numpy Representation of Features 

Dimension = (3, 4) # Rows, Columns

[[1,2,3,1],
[2,1,3,1],
[5,6,1,3]]

Representation of Weights

- If there is just a single neuron and all the inputs are passed to it

Dimension = (3, 1)

[[4],
[3],
[2]]

- If there are 2 neurons and all the inputs are passed to both of them

Dimension = (3, 2)

[[4,5],
[0,3],
[1,2]]

- If there are 3 neurons and all the inputs are passed to both of them

Dimension = (3, 3)

[[4,5,8],
[0,3,2],
[1,2,5]]

Here [4,0,1] are the weights for the first neuron, [5,3,2] are the weights for the second
neuron and [8,2,5] are the weights for the third neuron

To add more layers just compute the summation and multiply it with another weight matrix
where the dimensions should be -
(output of last layer/columns of last weight matrix, output of this layer/neurons in this layer)

The number of outputs for a certain layer depends on the number of neurons in that layer.

"""

import numpy as np

### LAYER 1
# samples -> 4, features -> 2
X1 = np.random.randn(4, 2)

# features, units for next layer
W1 = np.random.randn(2, 3)
summation1 = np.dot(X1, W1)

### LAYER 2
X2 = summation1
# Units of last layer, output units 
W2 = np.random.randn(3, 1)
summation2 = np.dot(X2, W2)

def step_function(x):
    return np.where(x > 0, 1, 0)

output = step_function(summation2)
