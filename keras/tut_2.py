#https://youtu.be/Boo6SmgmHuM DONE
#https://www.youtube.com/watch?v=dzoh8cfnvnI DONE
#https://www.youtube.com/watch?v=2f-NjDUvZIE DONE
#https://www.youtube.com/watch?v=km7pxKy4UHU
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
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()


cm_plot_labels = ["no_side_effects", "had_side_effects"]
plot_confusion_matrix(cm, cm_plot_labels, title="Confusion Matrix")
