import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

seed = 7
numpy.random.seed(seed) # SETUP SEED FOR R GEN

(X_train, y_train), (X_test, y_test) = mnist.load_data() # LOADS THE DATA

# FLATTENING IMAGES FROM A 3-D ARRAY TO 1-D ARRAY
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32') # CONVERT TO 32-BIT VALUES
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32') # CONVERT TO 32-BIT VALUES

# VALUES OF PIXELS ARE 0-255 IN GRAYSCALE, SO WE NORMALIZE TO 0-1
X_train = X_train/255
X_test = X_test/255

# ONE-HOT ENCODING VALUES (??)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# FUNCTION FOR A BASELINE MODEL
def baseline_model():
    model = Sequential() # CREATE A SEQUENTIAL MODEL
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu')) # LAYER 1
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax')) # LAYER 2

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = baseline_model()

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline error: %.2f%%" %(100-scores[1]*100))
