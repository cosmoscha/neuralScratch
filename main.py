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
#you MUST do np.dot(weights, inputs) + bias in that order
