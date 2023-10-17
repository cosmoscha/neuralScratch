import sys
import numpy as np
import matplotlib

inputs = [1, 2, 3, 2.5]
weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

weights = [weights1,
           weights2,
           weights3]
bias1 = 2
bias2 = 3
bias3 = 0.5

biases = [bias1,
        bias2,
        bias3]


# output = [inputs[0] * weights1[0] + inputs[1] * weights1[1] + inputs[2] * weights1[2] + inputs[3] * weights1[3] + bias1,
#           inputs[0] * weights2[0] + inputs[1] * weights2[1] + inputs[2] * weights2[2] + inputs[3] * weights2[3] + bias2,
#           inputs[0] * weights3[0] + inputs[1] * weights3[1] + inputs[2] * weights3[2] + inputs[3] * weights3[3] + bias3]
#inputs * weights + bias

layer_outputs = []
for neuron_weights, neuron_bias in zip(weights, biases):
    #zip combines two lists into a list of lists. so this is saying for each element within each list, do something. so for
    #weights1[0] and inputs[0]
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs)

#explain this code
#for each element in weights and biases, and for each element within inputs and weights, add to the output the input * the weight, add to output the bias. thats the result


inputs=[1, 2, 3, 2.5]
weights= [0.2 , 0.8, -0.4, 1.0]
bias = 2
output = np.dot(weights, inputs) + bias
print(output)

# np.dot() is just that above formula simplified
#you MUST do np.dot(weights, inputs) + bias in that order.. if they are 1D

#how to transpose incompatibe matrixes

inputs = [
    [1.2, 2.5, 3.7, 4.1],
    [5.9, 6.3, 7.6, 8.2],
    [9.4, 10.8, 11.5, 12.7]
]

weights = [
    [-1, -2, -3, -4],
    [-5, -6, -7, -8],
    [-9, -10, -11, -12]
]

biases = [2 , 3, 0.5]

out = np.dot(inputs, np.array(weights).T) + biases
print(out)

#you do transpose to weights, bc it is "shorter" than the inputs. so it becomes 
[    [-1, -5, -9],
    [-2, -6, -10],
    [-3, -7, -11],
    [-4, -8, -12]
]

# now it is as tall as inputs are wide, so you can multiply

#lets make a new layer

inputs = [
    [1.2, 2.5, 3.7, 4.1],
    [5.9, 6.3, 7.6, 8.2],
    [9.4, 10.8, 11.5, 12.7]
]

weights = [
    [-1, -2, -3, -4],
    [-5, -6, -7, -8],
    [-9, -10, -11, -12]
]

biases = [2 , 3, 0.5]

inputs1 = [[0.9, -0.6, 0.5, -0.3],
 [1.0, 0.0, -0.7, 0.1],
 [-0.5, 0.2, 0.9, -0.4]]



weights1 = [[-0.2, -0.9, -0.3, 0.8],
 [-0.8, 0.7, -0.6, -0.5],
 [-0.5, 0.3, -0.9, 0.7]]


biases1 = [2.8, -2.2, 1.4]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights1).T) + biases1
print(layer2_outputs)

# this is getting messy! so we should make it into an object
np.random.seed(0)
x  = [
    [1.2, 2.5, 3.7, 4.1],
    [5.9, 6.3, 7.6, 8.2],
    [9.4, 10.8, 11.5, 12.7]
]


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs, n_neurons)
        
        #how many layers do you want?

        self.biases = np.zeros((1, n_neurons))

        #the bias matrix, its 1 row by a number of neurons(inputs)
    def forward(self):
        self.outs = np.dot(inputs, self.weights) +self.biases
        #obviously this is just the formula we are familar with so far

    # this methods means we dont need to transpose anymore
    
layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)
#layer 2 is 5 rows tall, because layer 1 is 5 columns wide. they HAVE to match
layer1.forward(x)
layer2.forwards(layer1.output)
print("layer2 output", layer2.output)


print(np.random.randn(4,3))
#this will create a 4 x 3 matrix (4 rows by 3 columns)
#sometimes, the numbers are too big
print(0.10*np.random.randn(4,3))