#https://youtu.be/UkzhouEk6uY
import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler


# Example data of a drug test on 2100 participants, 13 - 100 yrs
# half <65, half >65
# 95% 65+ had side effects
# 95% <65 had no side effects

def generate_training_data():
	train_labels = []
	train_samples = []

	for i in range(50):
		# 5% of the younger population which experienced the side effects
		random_younger = randint(13, 64)
		train_samples.append(random_younger)
		train_labels.append(1)
		
		# 5% of the older population which hadn't experienced the side effects
		random_older = randint(65, 100)
		train_samples.append(random_older)
		train_labels.append(0)
		
	for i in range(1000):
		# 95% of the younger population which hadn't experienced the side effects
		random_younger = randint(13, 64)
		train_samples.append(random_younger)
		train_labels.append(0)
		
		# 95% of the older population which experienced the side effects
		random_older = randint(65, 100)
		train_samples.append(random_older)
		train_labels.append(1)
	
	train_labels = np.array(train_labels)
	train_samples = np.array(train_samples)
	
	# Scaling the samples
	scaler = MinMaxScaler(feature_range=(0,1)) # Scales from 0 to 1
	scaled_train_samples = scaler.fit_transform((train_samples).reshape(-1, 1)) # Because this fit transform function doesn't accept 1D arrays
																				# it's reshaped from -1 to 1
	return scaled_train_samples, train_labels

# GENERATING TEST DATA (REAL-WORLD DATA ON WHICH TO MAKE PREDICTION)
def generate_test_data():
	test_labels = []
	test_samples = []

	for i in range(10):
		random_younger = randint(13, 64)
		test_samples.append(random_younger)
		test_labels.append(1)
		
		random_older = randint(65, 100)
		test_samples.append(random_older)
		test_labels.append(0)
		
	for i in range(200):
		random_younger = randint(13, 64)
		test_samples.append(random_younger)
		test_labels.append(0)
		
		random_older = randint(65, 100)
		test_samples.append(random_older)
		test_labels.append(1)
		
		
	test_labels = np.array(test_labels)
	test_samples = np.array(test_samples)

	scaler = MinMaxScaler(feature_range=(0,1))
	scaled_test_samples = scaler.fit_transform((test_samples).reshape(-1,1))
	return scaled_test_samples, test_samples, test_labels

# Print the raw data
#print("Train samples:")
#for i in train_samples:
#	print(i, " "),
	


# Print scaled
#for i in scaled_train_samples:
#	print(i)
