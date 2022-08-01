# hidden layers activation functions
#  ReLU
import matplotlib.pyplot as plt

def relu(input):
    return max(0.0, input)

x = [i for i in range(-10, 10)]
y = [relu(i) for i in x]

plt.plot(x,y)
plt.show()

# sigmoid/logistic
from math import exp

def sigmoid(x):
    return 1/(1 + exp(-x))

x = [i for i in range(-10, 10)]
y = [sigmoid(i) for i in x]

plt.plot(x,y)
plt.show()


# tanh
def tanh(x): 
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

x = [i for i in range(-10, 10)]
y = [tanh(i) for i in x]

plt.plot(x,y)
plt.show()


# output layer activation functions

# linear/no activation/identity
def linear(x):
    return x


# softmax
def softmax(x):
    return exp(x) / exp(x).sum()

x = [1,2,3]
print(softmax(x).sum())
