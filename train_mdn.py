import csv
import numpy as np
from mixture_density_net import MixtureDensityNet


reader = csv.reader(open('test-distribution.csv', 'r'))
data = list(reader)

data = np.array(data, dtype=float)

x = data[1:100, 0:2]
t = data[1:100, 2]

net = MixtureDensityNet(2, 4)
net.train(x, t)
print(net.W2)
print(net.W1)