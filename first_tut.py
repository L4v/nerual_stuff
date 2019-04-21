#https://www.youtube.com/watch?v=kft1AJ9WVDk
#TODO https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
import numpy as np

# Sigmoid funkcija, sluzi za normalizaciju ulaza <=> Za ulaz s intervalom vrednosti [a, b] normalizuje se u [0, 1] (sigmmoid to radi)
# taj interval je podrzan od strane neuronske mreze
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_dt(x):
	return x * (1 - x)

# Ulaz za "treniranje", tj obuku NM
training_inputs = np.array([[0,0,1],
							[1,1,1],
							[1,0,1],
							[0,1,1]])

# Ocekivani izlazi na poznat ulaz (za treniranje)
training_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)

# Sinapticke tezine, vrednosti pomocu kojih se koriguje izlaz NM
synaptic_weights = 2 * np.random.random((3, 1)) - 1

print("Random starting synaptic weights: ")
print(synaptic_weights)

# Treniranje NM-a
for iteration in range(20000):
	input_layer = training_inputs # Postavljanje ulaza NM-a na zeljene vrednosti
	
	outputs = sigmoid(np.dot(input_layer, synaptic_weights)) # Izlazi iz NM-a
	
	error = training_outputs - outputs # Greska dobijenog izleza u odnosu na ocekivani
	
	adjustments = error * sigmoid_dt(outputs) # Vrednosti kojima se podesavaju sinapticke tezine
	
	synaptic_weights += np.dot(input_layer.T, adjustments) # Podesavanje sinaptickih tezina
	
print("Synaptic weights after training:")
print(synaptic_weights)
	
print("Outputs after training: ")
print(outputs)
