import sys
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()
np.random.seed(0)

X = [[1,2,3,2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5 , 2.7, 3.3, -0.8]]
X, y = spiral_data(100, 3) #y is number of classes
inputs = [0.2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

# for i in inputs:
#     if i > 0:
#         output.append(i)
#     elif i <= 0:
#         output.append(0)

for i in inputs:
    output.append(max(0,i))

print(output)
# "----------------------------------------------------------""
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self):
        self.outs = np.dot(inputs, self.weights) +self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.out = np.maximum(0, inputs)

    
# layer1 = Layer_Dense(4,5)
# layer2 = Layer_Dense(5,2)

# layer1.forward(X)
# layer2.forward(layer1.output)
# print(layer2.output)


layer1 = Layer_Dense(2,5)
activation1 = Activation_ReLU()
layer1.forward(X)
activation1.forward(layer1.output)
# print(activation1.output)


# "----------------------------------------------------------""
def spiral_data(points, classes): #x and y, how many inputs and neurons
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

# print("here")
X, y = spiral_data(100,3)

plt.scatter(X[: , 0], X[:,1], c=y, cmap="brg")
plt.show()

# "----------------------------------------------------------""

#SOFTMAX ACTIVATION

#first, exponentiate the outputs


layer_outputs = [4.8, 1.21, 2.385 ]
#E = 2.71828182846

E = math.e

exp_values = []
for output in layer_outputs:
    exp_values.append(E**output)
print(exp_values)

#this can be written more simply as 
exp_values = np.exp(layer_outputs)


#next, normalize the values

norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append(value / norm_base)

print(norm_values)

# or write it like this
norm_values = exp_values /np.sum(exp_values)

print(sum())


# "----------------------------------------------------------""

layer_outputs = [[4.8, 1.21, 2.385 ],
                 [8.9, -1.81, 0.2 ],
                 [1.41, 1.051, 0.026 ]]

exp_values = np.exp(layer_outputs)

print(np.sum(layer_outputs, axis=1, keepdims=True)) #sums up the columns when 0. sums up rows when 1. keepdims will keep the shape of the matrix. for example we are adding up the rows when axis is one, but we want it to be "tall" instead of "wide"

norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)
print(sum(norm_values))
print(exp_values)

#-------------------------------------------------
#class is back in session

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self):
        self.output = np.dot(inputs, self.weights) +self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print("this is the activation softmax",activation2.output[:5]) #first 5

# [[0.33333334 0.33333334 0.33333334]
#  [0.33331734 0.3333183  0.33336434]
#  [0.3332888  0.33329153 0.33341965]
#  [0.33325943 0.33326396 0.33347666]
#  [0.33323312 0.33323926 0.33352762]]

#---------------------------------------------------------------------

#confidence score

# e ** x = b, e is close to 2.71828 aka eulers number
# import numpy as np
# import math
# b = 5.2

# print(np.log(b))
# print(math.e ** 1.648658)
#e raised to what will equal to b? In this case it is 1.648658, because 2.71 ** 1.64 = 5.2

import math

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0]) * target_output[0] + 
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2])

print(loss)
loss = -math.log(softmax_output[0])
print(loss)

print(math.log(0.7))
print(math.log(0.5))
target_class = 0
#this means if we cet a result closer to the 0th index, it is "hot", and close to the target that we are trying to attain

#LOSS IS JUST THE OPPOSITE OF CONFIDENCE

#=-------------------------------------------

#implementing the actual loss:

import math

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]

