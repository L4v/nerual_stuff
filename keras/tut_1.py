#https://youtu.be/UkzhouEk6uY
import numpy as np
from random import randint
from sklern.preprocessing import MinMaxScalar

train_labels = []
train_samples = []

# Example data of a drug test on 2100 participants, 13 - 100 yrs
# half <65, half >65
# 95% 65+ had side effects
# 95% <65 had no side effects

for i in range(1000):
	random_younger = randint(13, 64)
	train_samples.append(random_younger)
	train_labels.append(0)
	
	random_older = randint(65, 100)
	train_samples.append(random_older)
	train_labels.append(1)
	
for i in range(50):
	random_younger = randint(13, 64)
	train_samples.append(random_younger)
	train_labels.append(1)
	
	random_older = randint(65, 100)
	train_samples.append(random_older)
	train_labels.append(0)
	

# Print the raw data
print("Train samples:")
for i in train_samples:
	print(i + " "),
	
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)

# Scaling the samples
scaler = MinMaxScaler(feature_range(0,1)) # Scales from 0 to 1
scaled_train_samples = scaler.fit_transform((train_samples).reshape(-1, 1)) # Because this fit transform function doesn't accept 1D arrays
																			# it's reshaped from -1 to 1

# Print scaled
for i in scaled_train_samples:
	print(i)
