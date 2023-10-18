TIME TO LEARN ACTIVATION!

evey neuron has an activation function

we need this, because data is never linear

a rectified linear function is better, because it accounts for curvature of dataset

if x is greater than x, x is x. if x is less than 0, it is 0
thats about it as far as activation goes, but it is very important because this is how the data is adjusted to the expected result

now that we know how to make the neural network, its now time to learn how to train it
so we need to find out, how wrong is the result?
in comes softmax activation. This will optimize so you dont get ONLY absolute value. Not everything is a yes or no, you need to know HOW yes or HOW no it is

now that we know how to do softmax, now we can move on to how do we finetune it even more so that it matches the curve. NOW its time for backpropagation, for accuracy

how confident is it? 100? 90? thats why we need to know cross entropy and mean absolute error
How wrong is the model? categorial cross-entropy is the default method to measure mean absolute error


one-hot encoding

classes 2
label: 0
one-hot: [1,0]

classes 5
label: 1
one-hot: [0,1,0,0,0]

classes 5
label: 3
one-hot: [0,0,0,1,0]

and etc

lets discuss what is a log function? generally it is 'solving for x'

where e ** x = b

#optimization

you've calculated your inputs, know how to shape them, you know how to activate and refine, but you need to know how to adjust the weights and biases in realtime. Which is the final step, optimization.

we dont want to make random combinations until something fits, that would take forever

