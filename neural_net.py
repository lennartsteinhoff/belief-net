from numpy import array, exp, random, dot


def sigmoid(x):
    return 1 / (1 + exp(-x))


def sigmoid_derivate(x):
    return sigmoid(x) * (1-sigmoid(x))


class NeuralNet:

    def __init__(self):
        self.input_size = 2
        self.hidden_size = 3
        self.output_size = 1

        self.W1 = random.randn(self.hidden_size, self.input_size)
        self.W2 = random.randn(self.output_size, self.hidden_size)

        # self.z1 = array([[0], [0], [0]])
        # self.h1 = array([[0], [0], [0]])
        # self.z2 = array([0])
        # self.h2 = array([0])

    def forward(self, x):
        self.a1 = self.W1.dot(x.T)
        self.z1 = sigmoid(self.a1)
        self.a2 = self.W2.dot(self.z1)
        self.z2 = sigmoid(self.a2)
        return self.z2.T

    def backward(self, x, t, y):
        self.y_error = (t - y).T
        self.y_delta = self.y_error * sigmoid_derivate(y.T)

        # self.h1_error = self.o_delta.dot(self.W2.T)
        self.z1_error = self.W2.T.dot(self.y_error)
        self.z1_delta = self.z1_error * sigmoid_derivate(self.z1)

        self.W2 += 0.001 * self.z1.dot(self.y_delta.T).T
        self.W1 += x.T.dot(self.z1_delta.T).T


    def train(self, x, t):
        y = self.forward(x)
        self.backward(x, t, y)




for count in range(10000):
    neural_net.train(x_simple, t_simple)