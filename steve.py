import numpy as np
from steve_mind import NeuralNetwork
if __name__ == "__main__":
	
	neural_network = NeuralNetwork()
	
	neural_network.setup_weights(2 * np.random.random((3, 1)) - 1)
	
	print ("Random synaptic weights:")
	print(neural_network.synaptic_weights)
	
		# Ulaz za "treniranje", tj obuku NM
	training_inputs = np.array([[0,0,1],
								[1,1,1],
								[1,0,1],
								[0,1,1]])

	# Ocekivani izlazi na poznat ulaz (za treniranje)
	training_outputs = np.array([[0,1,1,0]]).T
	
	neural_network.train(training_inputs, training_outputs, int(input("Number of training iter.: ")))
	
	print("Synaptic weights after training:")
	print(neural_network.synaptic_weights)
	
	A = str(input("Input 1: "))
	B = str(input("Input 2: "))
	C = str(input("Input 3: "))
	
	print("New situation: input_data = ", A, B, C)
	print("Output data:")
	print(neural_network.think(np.array([A, B, C])))
