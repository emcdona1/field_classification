import os
import argparse
import time
import random
import cv2
import numpy as np
from numpy.random import seed
import matplotlib
matplotlib.use('Agg') # required when running on Vortex server
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras import regularizers
from keras.models import model_from_json
from keras.models import load_model
from scipy import interp
from sklearn.metrics import roc_curve, confusion_matrix, auc
from sklearn.model_selection import StratifiedKFold

# setup
SEED = 1
seed(SEED)
tf.compat.v1.random.set_random_seed(SEED)
random.seed(SEED)
BATCH_SIZE = 64 * 32
LEARNING_RATE = 0.0001 * 32

def build_model(img_size): # create model architecture and compile it # change so all of the parameters are passed in
	""" Creates layers for model and compiles model.
	Parameters:
	-----
	@ img_size : int
	Dimensions of images being inputted (img_size x img_size in pixels)

	Output:
	-----
	@ model : keras.Sequential
	"""
	model = Sequential()

	# Image input shape: 256 x 256 x 3

	# 1. Convolution Layer: 10 filters of 5px by 5px
	model.add(Conv2D(10, (5, 5), input_shape = (img_size, img_size, 3))) 
	# Output shape: 10 x 252 x 252

	# 2. Batch Normalization: Normalizes previous layer to have mean near 0 and S.D. near 1
	model.add(BatchNormalization())
	# Output shape: 10 x 252 x 252

	# 3. Activation Layer: ReLU uses the formula of f(x)= x if x>0 and 0 if x<=0
	# Apparently it's a pretty common one for CNN so we're going with the flow here
	model.add(Activation("relu"))
	# Output shape: 10 x 252 x 252

	# 4. Pooling function: from the paper, it didn't specify function, but looking online, it seems that the default is Max so we are a-okay here
	model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
	# Output shape: 10 x 126 x 126

	#-------------Next Set of Layers--------------
	# 5. Convolution Layer: 40 filters of 5px by 5px
	model.add(Conv2D(40, (5, 5)))
	# Output shape: 40 x 122 x 122

	# 6. Batch Normalization Layer
	model.add(BatchNormalization())
	# Output shape: 40 x 122 x 122

	# 7. Activation Layer: Same as above
	model.add(Activation("relu"))
	# Output shape: 40 x 122 x 122

	# 8. Pooling again will decrease "image shape" by half since stride = 2
	model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
	# Output shape: 40 x 61 x 61

	# ----------Hidden layers-----------

	# 9. Flattening Layer: Make pooled layers (that look like stacks of grids) into one "column" to feed into ANN
	model.add(Flatten())

	# 10. Dropout Layer: In Mathematica Dropout[] has a rate of dropping 50% of elements and multiply rest by 2
	# !!!!!!! Currently trying to figure out how to do the multiply by 2 but moving on for now !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	model.add(Dropout(0.5, seed=SEED))

	model.add(Dense(500, activation = "linear", \
				activity_regularizer = regularizers.l2(0.01), \
				kernel_regularizer = regularizers.l2(0.05))) # kernel_regularizer=regularizers.l2(0.1)))
	model.add(Dense(500, activation = "relu", \
				activity_regularizer = regularizers.l2(0.01), \
				kernel_regularizer = regularizers.l2(0.05))) 
	#model.add(Activation("relu"))

	model.add(Dropout(0.25, seed = SEED))
	# The output layer with 2 neurons, for 2 classes
	model.add(Dense(2, activation = "linear", \
				activity_regularizer = regularizers.l2(0.01), \
				kernel_regularizer = regularizers.l2(0.05)))
	model.add(Dense(2, activation = "softmax", \
				activity_regularizer = regularizers.l2(0.01), \
				kernel_regularizer = regularizers.l2(0.05)))
	# model.add(Activation("softmax"))

	opt = tf.keras.optimizers.Adam(lr = LEARNING_RATE, beta_1 = 0.9, beta_2 = 0.999, \
		epsilon = 0.00001, decay = 0.0, amsgrad = False)
	model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
	return model

def plot_accuracy_and_loss(history, index):
	''' Create plots of accuracy and loss, save to disk.

	Parameters:
	-----
	@ history : History
	History object of results, returned from the model.fit() method.

	@ index : int
	Current fold.

	Output:
	-----
	none (file output)

	Two PNG files saved in graphs folder
	'''
	# Save a graph of the testing/training accuracy during the training phase
	plt.figure(1)
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.savefig(os.path.join('graphs', 'val_accuracy_' + str(index + 1) + '.png'))
	plt.clf()

	# Save a graph of the testing/training loss during the training phase
	plt.figure(2)
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc = 'upper left')
	plt.savefig(os.path.join('graphs', 'val_loss_' + str(index + 1) + '.png'))
	plt.clf()

def plot_ROC_for_Kfold(mean_fpr, mean_tpr, mean_auc, std_auc):
	""" Update and save mean ROC plot after each fold.

	Parameters:
	------
	@ mean_fpr : float
	false positive rate (mean from all folds run so far)

	@ mean_tpr : float
	true postive rate (mean from all folds run so far)

	@ mean_auc : float
	area under ROC curve (mean from all folds run so far)

	@ std_auc : float
	standard deviation of AUC (mean from all folds run so far)

	Output:
	------
	none

	Saves plot as `mean_ROC.png` in graphs folder.
	"""
	plt.figure(3)
	plt.plot([0, 1], [0, 1], linestyle = '--', lw = 2,
				color = 'r', label='Random Chance', alpha = 0.8)
	plt.plot(mean_fpr, mean_tpr, color = 'blue',
				label = r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
				lw = 2, alpha = 0.8)
	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic (ROC) curve')
	plt.legend(loc="lower right")
	plt.savefig(os.path.join('graphs', 'mean_ROC.png'))
	plt.clf()

def import_images(img_directory, folders, img_size): 
	""" Import images from the file system and returns two numpy arrays containing the pixel information and classification.

	Parameters:
	-----
	@ img_directory : String
	Directory which contains the image folders

	@ folders : String list of length = 2
	Names of folders containing images (images must be in separate folders by species)
	
	@ img_size : int
	Pixel dimensions of images

	Output:
	-----
	@ features : numpy arrays
	Contains RGB values for each image
	(dimensions: # of images x image width x image height x 3)
	
	@ labels : numpy array
	Contains the class label (0/1) of the corresponding image
	(# of rows = # of images, # of columns = 1)
	"""
	all_data = []
	for category in folders:
		path=os.path.join(img_directory, category) #look at each folder of images
		class_index = folders.index(category)
		for img in os.listdir(path): # look at each image
			try:
				img_array = cv2.imread(os.path.join(path,img), -1) #-1 means image is read as color
				img_array = img_array/255.0
				all_data.append([img_array, class_index]) #, img])
			except Exception as e:
				pass
	random.shuffle(all_data)
	print("Loaded and shuffled data")

	features = []
	labels = []
	img_names = []

	#store the image features (array of RGB for each pixel) and labels into corresponding arrays
	for data_feature, data_label in all_data:
		features.append(data_feature)
		labels.append(data_label)
		# img_names.append(file_name)

	#reshape into numpy array
	features = np.array(features) #turns list into a numpy array
	features = features.reshape(-1, img_size, img_size, 3) # 3 bc three channels for RGB values
		# -1 means "numpy figure out this dimension," so the new nparray has the dimensions of: [#_of_images rows, img_size, img_size, 3] 
	labels = np.array(labels)
	return [features,labels]

def train_cross_validate(n_folds, img_directory, folders, img_size, num_epochs):
	""" Import images from the file system and returns two numpy arrays containing the pixel information and classification.

	Parameters:
	-----
	@ n_folds : int
	Number of folds to train on (minimum of 2)

	@ img_directory : String
	Directory which contains the image folders

	@ folders : String list of length = 2
	Names of folders containing images (images must be in separate folders by species)
	
	@ img_size : int
	Pixel dimensions of images

	@ num_epochs : int
	Maximum number of epochs to perform on each k-fold.

	Output:
	-----
	none
	"""
	# initialize stratifying k fold
	skf = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = SEED)

	# data frame to save values of loss and validation after each fold
	results = pd.DataFrame()
	#obtain images
	data = import_images(img_directory, folders, img_size)
	features = data[0]
	labels = data[1]
	print("Stored features and labels")
	# for roc plotting
	tprs = []
	aucs = []
	mean_fpr = np.linspace(0, 1, 100)
	cm_file = open(os.path.join('graphs', 'confusion_matrix.txt'), 'w')

	for index, (train_indices, val_indices) in enumerate(skf.split(features, labels)):
		print("Training on fold " + str(index + 1) + "/" + str(n_folds))
		train_features = features[train_indices]
		train_labels = labels[train_indices]
		print("Training data obtained")
		val_features = features[val_indices]
		val_labels = labels[val_indices]
		print("Validation data obtained")

		# Create new model each time
		model = None
		model = build_model(img_size)
		print("Training model")
		es_callback = EarlyStopping(monitor = 'val_loss', patience = 4, restore_best_weights = True)
		history = model.fit(train_features, train_labels, \
			batch_size = BATCH_SIZE, epochs = num_epochs, \
			callbacks = [es_callback], validation_data = (val_features, val_labels))

		model.save('saved_models/CNN_' + str(index + 1) + '.model')
		
		plot_accuracy_and_loss(history, index)

		# Compute ROC curve and area the curve
		probas = model.predict_proba(val_features)[:,1] # 0 = definitely c1, 1 = definitely c2
		prob_classification = [round(a + 0.001) for a in probas]
		# Compute ROC curve and area the curve
		fpr, tpr, thresh = roc_curve(val_labels, probas)
		tn, fp, fn, tp = confusion_matrix(val_labels, prob_classification).ravel()
		confusion_mat = '\t\t   Predicted\n\t\t  P\t    N\t\n\t\t  --------------\n\t\tP|  ' + str(tp)
		confusion_mat += ' \t|  ' + str(fn)
		confusion_mat += ' \t|\nActual\t  --------------\n\t\tN|  ' + str(fp)
		confusion_mat += ' \t|  ' + str(tn)
		confusion_mat += ' \t|\n\t\t  --------------\n'
		print(confusion_mat)
		cm_file.write(confusion_mat)

		tprs.append(interp(mean_fpr, fpr, tpr))
		tprs[-1][0] = 0.0
		roc_auc = auc(fpr, tpr)
		aucs.append(roc_auc)
		# use the mean statistics to compare each model (that we train/test using 10-fold cv)
		mean_tpr = np.mean(tprs, axis = 0)
		mean_tpr[-1] = 1.0
		mean_auc = auc(mean_fpr, mean_tpr)
		std_auc = np.std(aucs)

		# plot the mean ROC curve and display AUC (mean/st dev)
		plot_ROC_for_Kfold(mean_fpr, mean_tpr, mean_auc, std_auc)
		
		len_history = len(history.history['loss'])
		results = results.append([[index + 1, \
			history.history['loss'][len_history - 1], \
			history.history['acc'][len_history - 1], \
			history.history['val_loss'][len_history - 1], \
			history.history['val_acc'][len_history - 1], \
			tn, fp, fn, tp]])
	
	results = results.rename({0: 'Fold Number',\
					1: 'Training Loss',\
					2: 'Training Accuracy',\
					3: 'Validation Loss', \
					4: 'Validation Accuracy',\
					5: 'True Negatives', 6: 'False Positives', \
					7: 'False Negatives', 8: 'True Positives'})
	results.to_csv(os.path.join('graphs','final_acc_loss.csv'), encoding='utf-8', index=False)
	cm_file.close()
	
	
if __name__ == '__main__':
	start_time = time.time()
	parser = argparse.ArgumentParser('import images and train model')
	parser.add_argument('-d', '--directory', default='', help='Folder holding category folders')	
	parser.add_argument('-c1', '--category1', default='lyco_train', help='Folder of class 1')
	parser.add_argument('-c2', '--category2', default='sela_train', help='Folder of class 2')
	parser.add_argument('-s', '--img_size', default=256, help='Image dimension in pixels')
	parser.add_argument('-n', '--number_folds', default=10, help='Number of folds (minimum 2) for cross validation')
	parser.add_argument('-e', '--number_epochs', default=25, help='Number of epochs')

	args = parser.parse_args()

	img_directory = args.directory
	folders = [args.category1, args.category2]
	img_size = int(args.img_size)
	n_folds = int(args.number_folds)
	n_epochs = int(args.number_epochs)

	
	if not os.path.exists('graphs'):
		os.makedirs('graphs')
	if not os.path.exists('saved_models'):
		os.makedirs('saved_models')
	
	train_cross_validate(n_folds, img_directory, folders, img_size, n_epochs)
	print('c1: ' + folders[0] + ', c2: ' + folders[1])
	end_time = time.time()
	print('Completed in %.1f seconds' % (end_time - start_time))