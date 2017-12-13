from numpy import random, exp, sqrt, pi, exp, square, log, power
import numpy as np

def sigmoid(x):
    return 1 / (1 + exp(-x))


def sigmoid_dot(x):
    return sigmoid(x) * (1 - sigmoid(x))

def p(t, mu, sigma):
    return exp(square(t - mu) / (-2 * square(sigma))) / (sqrt(2 * pi * square(sigma)))

def error_function(t, mu, sigma):
    return - log(p(t, mu, sigma))

def d_error_function(t, mu, sigma):
    d_e_mu = (mu - t) / square(sigma)
    d_e_sigma = - ((square(t - mu) / power(sigma, 3)) - 1 / sigma)
    return np.array([d_e_mu, d_e_sigma]).T

def d_error_function_numeric(t, mu, sigma, delta):
    d_e_mu = (error_function(t, mu + delta, sigma) - error_function(t, mu, sigma)) / delta
    d_e_sigma = (error_function(t, mu, sigma + delta) - error_function(t, mu, sigma)) / delta
    return np.array([d_e_mu, d_e_sigma]).T

class MixtureDensityNet:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 2

        self.W1 = random.randn(self.hidden_size, self.input_size)
        self.W2 = random.randn(self.output_size, self.hidden_size)

    def forward(self, x):
        a1 = self.W1.dot(x.T)
        z1 = sigmoid(a1)
        a2 = self.W2.dot(z1)
        z2 = a2
        return z2.T

    def likelihood(self, x, t):
        phi = self.forward(x)

        values = zip(t, phi)
        likelihood = array([])
        for t, mu, sigma in values:
            likelihood.append(p(t, mu, sigma))

        return likelihood

    def error(self, X):


