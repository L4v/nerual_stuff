#https://youtu.be/Boo6SmgmHuM DONE
#https://www.youtube.com/watch?v=dzoh8cfnvnI DONE
#https://www.youtube.com/watch?v=2f-NjDUvZIE
import keras
from tut_1 import scaled_train_samples, train_labels
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

model.compile(Adam(lr=.001), loss = 'sparse_categorical_crossentropy', metrics=['accuracy']) # Adam - optimization function(gradient descent, etc..), lr - learning rate

																							 # loss - loss function (mean square error, etc...), metrics - on what to judge performance of model
model.fit(scaled_train_samples, train_labels, validation_split=0.1, # Training, samples to train on, labels for the samples, validation_data will be 10% of the training samples
		 batch_size=10, epochs=20, shuffle=True, verbose=2) # batch size, epoch - how many runs


