from numpy import random, exp, sqrt, pi, exp, square, log, power
import numpy as np


def sigmoid(x):
    return 1 / (1 + exp(-x))


def sigmoid_dot(x):
    return sigmoid(x) * (1 - sigmoid(x))


class ErrorFunction:

    def __init__(self, delta):
        self.delta = delta

    def p(self,t, mu, sigma):
        return exp(square(t - mu) / (-2 * square(sigma))) / (sqrt(2 * pi * square(sigma)))

    def values(self,t, mu, sigma):
        return - log(self.p(t, mu, sigma))

    def sum(self,t, mu, sigma):
        return np.sum(self.values(t, mu, sigma))

    def derivative(self, t, mu, sigma):
        d_e_mu = (mu - t) / square(sigma)
        d_e_sigma = - ((square(t - mu) / power(sigma, 3)) - 1 / sigma)
        return np.array([d_e_mu, d_e_sigma]).T

    def numeric_derivative(self, t, mu, sigma):
        delta = self.delta
        d_e_mu = (self.values(t, mu + delta, sigma) - self.values(t, mu, sigma)) / delta
        d_e_sigma = (self.values(t, mu, sigma + delta) - self.values(t, mu, sigma)) / delta
        return np.array([d_e_mu, d_e_sigma]).T


class MixtureDensityNet:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 2

        self.W1 = random.randn(self.hidden_size, self.input_size)
        self.W2 = random.randn(self.output_size, self.hidden_size)

        self.error = ErrorFunction(delta=.01)

    def forward(self, x):
        self.a1 = self.W1.dot(x.T).T
        self.z1 = sigmoid(self.a1)
        self.a2 = self.W2.dot(self.z1.T).T
        self.z2 = self.a2
        return self.z2

    def backward(self, x, t, mu, sigma):
        self.y_delta = self.error.numeric_derivative(t, mu, sigma)[0]

        # self.h1_error = self.o_delta.dot(self.W2.T)
        self.z1_error = self.W2.T.dot(self.y_delta.T).T
        self.z1_delta = self.z1_error * sigmoid_dot(self.z1)

        self.W2 -= 0.001 * self.z1.T.dot(self.y_delta).T
        self.W1 -= 0.001 * x.T.dot(self.z1_delta).T

    def train(self, x, t):

        for count in range(100):
            y = self.forward(x)
            mu = np.array([y[:, 0]]).T
            sigma = np.array([y[:, 1]]).T
            print(self.error.sum(t, mu, sigma))
            print(np.min(sigma))
            self.backward(x, t, mu, sigma)




