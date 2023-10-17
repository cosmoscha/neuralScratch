refer to the img if necessary

the inputs are the information that is being passed through each neuron
the weights is how important those inputs are
the bias is the neurons tendency to lean one way or the other

what if you have an uneven amount of inputs and neurons? for example 4 inputs and 3 neurons

inputs = [1, 2, 3, 2.5]
weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]
bias1 = 2
bias2 = 3
bias3 = 0.5

output = [inputs[0] * weights1[0] + inputs[1] * weights1[1] + inputs[2] * weights1[2] + inputs[3] * weights1[3] + bias1,
          inputs[0] * weights2[0] + inputs[1] * weights2[1] + inputs[2] * weights2[2] + inputs[3] * weights2[3] + bias2,
          inputs[0] * weights3[0] + inputs[1] * weights3[1] + inputs[2] * weights3[2] + inputs[3] * weights3[3] + bias3]

this shows that the inputs are independent of each neuron, however how each neuron calculates weight for each input is different and each neuron has a different inherent bias

so its like input is a pokemon attack, the weight is the type advantage or disadvantage of the move vs the pokemon, and the bias is the pokemon's EVs and IVs that factor into their stats, that cannot be 
changed during the turn.

weights and biases are the knobs to turn to finetune the data when they are calculated

what is shape? 

2d array
lol = [[1,2,3,4],                   shape:
       [1,2,3,4]]                   (2, 4)

3d array
lol = [                             shape:
    [[1,2,3,4],                     (3, 2, 4)
    [1,2,3,4]],
    [[1,2,3,4],
    [1,2,3,4]]     
       ]


what is a tensor? an object that can be represented as an array

so in example 
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

inputs are vectors (2d), weights are matrices (3d)


what is dot product? 
a = [1,2,3]
b = [3,4,5]

dotproduct = a[0]*b[0] + a[1]*b[1]... etc etc 

batches are how we do input data. typically a 2d array
batch = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20],
    [21, 22, 23, 24],
    [25, 26, 27, 28],
    [29, 30, 31, 32]
]

shape(8,4)

oh it gets deeper. what is a matrix product?

its a matrix times matrix, to make a 3d matrix from the 2d one. Im not even going to type this one, you get the idea

sometimes two matrix do not have compative dimensions. for example a 4 x 3 matrix and a 3 x 2 matrix. 4 != 3. so we need to transpose
