#https://www.youtube.com/watch?v=PO4hePKWIGQ
from neural_network import NeuralNetwork
#import numpy as np
import matplotlib.pyplot as plot
from sklearn.datasets import load_digits
from sklearn import ensemble
import random

digits = load_digits() # Gets digits from database

images_and_labels = list(zip(digits.images, digits.target)) # Images with their desired values
plot.figure(figsize=(5,5)) 

# Plots the first 15 digits
#for index, (image, label) in enumerate(images_and_labels[:15]):
#	plot.subplot(3, 5, index + 1)
#	plot.axis("off")
#	plot.imshow(image, cmap=plot.cm.gray_r, interpolation = "nearest")
#	plot.title("%i" % label)
#plot.show()

#Define variables
n_samples = len(digits.images)
x = digits.images.reshape((n_samples, -1))
y = digits.target

#Create random indices 
sample_index=random.sample(range(len(x)),len(x)/5) #20-80
valid_index=[i for i in range(len(x)) if i not in sample_index]

#Sample and validation images
sample_images=[x[i] for i in sample_index]
valid_images=[x[i] for i in valid_index]

#Sample and validation targets
sample_target=[y[i] for i in sample_index]
valid_target=[y[i] for i in valid_index]

#Using the Random Forest Classifier
classifier = ensemble.RandomForestClassifier()

#Fit model with sample data
classifier.fit(sample_images, sample_target)

#Attempt to predict validation data
score=classifier.score(valid_images, valid_target)
print('Random Tree Classifier:\n') 
print ('Score\t'+str(score))

i=150

pl.gray() 
pl.matshow(digits.images[i]) 
pl.show() 
classifier.predict(x[i])
