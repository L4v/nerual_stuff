from neural_network import NeuralNetwork
import numpy as np
import matplotlib as plot

if __name__ == "__main()__":
	main()


def main():
	nn = NeuralNetwork()
	
	W = {}
	b = {}
	for l in range(1, len(nn_structure)):
		W[l] = np.random.random_sample((nn_structure[l], nn_structure[l-1]))
		b[l] = r.random_sample((nn_structure[l],))
	
	
	nn.setup_weights(weights);
