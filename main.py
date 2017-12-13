# simple example for a neural net
# taken and tested from
# https://dev.to/shamdasani/build-a-flexible-neural-network-with-backpropagation-in-python


from neural_net import NeuralNet

from numpy import array, exp, random, dot


def sigmoid(x):
    return 1 / (1 + exp(-x))


def sigmoid_derivate(x):
    return sigmoid(x) * (1-sigmoid(x))


x = array(([2, 9], [1, 5], [3, 6], [4, 4]), dtype=float)
t = array(([0.92], [0.86], [0.89], [0.90]))

W1 = random.randn(3, 2)
W2 = random.randn(1, 3)

a1 = W1.dot(x.T)
z1 = sigmoid(a1)
a2 = W2.dot(z1)
z2 = sigmoid(a2)

y = z2.T

y_error = (t - y).T
y_delta = y_error * sigmoid_derivate(y.T)

W2 += z1.dot(y_delta.T).T

z1_error = W2.T.dot(y_error)
z1_delta = z1_error * sigmoid_derivate(z1)

W1 += x.T.dot(z1_delta.T).T



neural_net = NeuralNet()
y = neural_net.forward(x)


neural_net.backward(x, t, y)

print("Predicted Output: " + str(y))
print("Acutal Output: " + str(t))


for b in range(10):

    neural_net.train(x, t)


y = neural_net.forward(x)
print("Predicted Output: " + str(y))
print("Acutal Output: " + str(t))