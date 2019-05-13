#https://youtu.be/Boo6SmgmHuM DONE
#https://www.youtube.com/watch?v=dzoh8cfnvnI DONE
#https://www.youtube.com/watch?v=2f-NjDUvZIE DONE
#https://www.youtube.com/watch?v=km7pxKy4UHU DONE
#https://www.youtube.com/watch?v=7n1SpeudvAE
import os; os.environ['KERAS_BACKEND'] = 'theano' # TO SET THEANO AS DEFAULT IF NOT CONFIG-ed ON THE OS
import keras
from tut_1 import generate_test_data, generate_training_data
from keras import Sequential
from random import randint
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import numpy as np

model = Sequential([
	Dense(16, input_shape=(1,), activation="relu"), # Dense input layer with 16 neurons, 1-D data nad RELU activation func.
	Dense(32, activation="relu"), # Dense hidden layer, 32 neurons, RELU activation func.
	Dense(2, activation="softmax") # Dense output layer, 2 neurons(true | false), SoftMax activation func
])
#model.add(l4)

model.summary() # Prints the model visualization

model.compile(Adam(lr=.001), loss = 'sparse_categorical_crossentropy', metrics=['accuracy']) # Adam - optimization function(gradient descent, etc..), lr - learning rate

																							 # loss - loss function (mean square error, etc...), metrics - on what to judge performance of model

training_data = generate_training_data()
scaled_train_samples = training_data[0]
train_labels = training_data[1]

model.fit(scaled_train_samples, train_labels, validation_split=0.1, # Training, samples to train on, labels for the samples, validation_data will be 10% of the training samples
		 batch_size=10, epochs=20, shuffle=True, verbose=2) # batch size, epoch - how many runs


# GENERATING TEST DATA (REAL-WORLD DATA ON WHICH TO MAKE PREDICTION)

scaled_data = generate_test_data()
scaled_test_samples = scaled_data[0]
test_samples = scaled_data[1]
test_labels = scaled_data[2]

# PREDICTION

predictions = model.predict(scaled_test_samples, batch_size=10, verbose=0)

print("MAKING PREDICTIONS:\nNO REACTION | REACTION")
for i in predictions:
	print(i)

rounded_predictions = model.predict_classes(scaled_test_samples, batch_size=10, verbose=0)
print("MAKING ROUNDED PREDICTIONS:")
for i in rounded_predictions:
	print(i, end = " ")
print("")


# CONFUSION MATRIX
cm = confusion_matrix(test_labels, rounded_predictions)
# FROM https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
							normalize=False,
							title="Connfusion matrix",
							cmap=plt.cm.Blues):

	plt.imshow(cm, interpolation="nearest", cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	if normalize:
		cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print("Confusion matrix, without normalization")

	print(cm)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j],
				horizontalalignment="center",
				color="white" if cm[i, j] > thresh else "black")

		plt.tight_layout()
		plt.ylabel("True label")
		plt.xlabel("Predicted label")


cm_plot_labels = ["no_side_effects", "had_side_effects"]
plot_confusion_matrix(cm, cm_plot_labels, title="Confusion Matrix")
plt.show()

# Saves all the model properties, weights, optimizer, states etc...
model.save("medical_trial_model.h5")

# json_string = model.to_json() # Saves only the model architecture as JSON

# model.save_weights("weights.h5") # Saves only the weights of the model
