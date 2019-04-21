from neural_network import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plot
from sklearn.datasets import load_digits

digits = load_digits()

plot.gray()
plot.matshow(digits.images[1])
plot.show()
