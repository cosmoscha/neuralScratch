import numpy as np

# Sigmoid Function (Logistic):
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Hyperbolic Tangent Function (Tanh):
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# Rectified Linear Unit (ReLU):
def relu(x):
    return np.maximum(0, x)

# Leaky ReLU:
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# Parametric ReLU (PReLU)
def prelu(x, alpha):
    return np.where(x > 0, x, alpha * x)

# Exponential Linear Unit (ELU)
def elu(x, alpha):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

# Scaled Exponential Linear Unit (SELU)
def selu(x, alpha, lambda_):
    return lambda_ * np.where(x > 0, x, alpha * (np.exp(x) - 1))

# Swish
def swish(x, beta):
    sigmoid_beta_x = 1 / (1 + np.exp(-beta * x))
    return x * sigmoid_beta_x

###SOFTMAX FUNCTION: used in output layer to adjust for multi class classificatgion problems


z = [2.0, 1.0, 0.1]


def softmax(z):
    e_z = np.exp(z)
    return e_z / e_z.sum()
