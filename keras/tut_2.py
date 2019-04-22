#https://youtu.be/Boo6SmgmHuM
import keras
from keras import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

model = Sequential([
	Dense(16, input_shape=(1,), activation="relu"), # Dense input layer with 16 neurons, 1-D data nad RELU activation func.
	Dense(32, activation="relu"), # Dense hidden layer, 32 neurons, RELU activation func.
	Dense(2, activation="softmax") # Dense output layer, 2 neurons(true | false), SoftMax activation func
])
#model.add(l4)

model.summary() # Prints the model visualization
