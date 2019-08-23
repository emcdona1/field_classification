# --- CONSTANTS ---
SEED = 1
# imports for reproducible results
from numpy.random import seed
seed(SEED)
from tensorflow.compat.v1 import set_random_seed
set_random_seed(SEED)
import random
random.seed(SEED)

# the ML stuff
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from keras.models import model_from_json
from keras.models import load_model
from keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

import cv2
import numpy as np
import os
import argparse
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import pandas as pd


def build_model(img_size): # create model architecture and compile it
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
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
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
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	# Output shape: 40 x 61 x 61

	# ----------Hidden layers-----------

	# 9. Flattening Layer: Make pooled layers (that look like stacks of grids) into one "column" to feed into ANN
	model.add(Flatten())

	# 10. Dropout Layer: In Mathematica Dropout[] has a rate of dropping 50% of elements and multiply rest by 2
	# !!!!!!! Currently trying to figure out how to do the multiply by 2 but moving on for now !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	model.add(Dropout(0.5, seed=SEED))

	model.add(Dense(500, activation="linear", activity_regularizer=regularizers.l2(0.01), kernel_regularizer=regularizers.l2(0.05))) # kernel_regularizer=regularizers.l2(0.1)))
	model.add(Dense(500, activation="relu", activity_regularizer=regularizers.l2(0.01), kernel_regularizer=regularizers.l2(0.05))) 
	#model.add(Activation("relu"))

	model.add(Dropout(0.25, seed=SEED))
	# The output layer with 2 neurons, for 2 classes
	model.add(Dense(2, activation="linear", activity_regularizer=regularizers.l2(0.01), kernel_regularizer=regularizers.l2(0.05)))
	model.add(Dense(2, activation="softmax", activity_regularizer=regularizers.l2(0.01), kernel_regularizer=regularizers.l2(0.05)))
	# model.add(Activation("softmax"))

	opt = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=0.00001, decay=0.0, amsgrad=False)
	model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
	return model

def plotROCforKfold(mean_fpr, mean_tpr, mean_auc, std_auc):
	plt.plot([0, 1], [0, 1], linestyle='--', lw=2,
				color='r', label='Random Chance', alpha=.8)
	plt.plot(mean_fpr, mean_tpr, color='blue',
				label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
				lw=2, alpha=.8)
	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.savefig('graphs/mean_ROC.png')
	plt.clf()
	# plt.show()

def import_images(data_dir, categories, image_size): 
	all_data = []
	for category in categories:
		path=os.path.join(data_dir,category) #look at each folder of images
		class_index = categories.index(category)
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
	features = np.array(features).reshape(-1, image_size, image_size, 3) #3 bc three channels for RGB values
	labels = np.array(labels)
	return [features,labels]


def train_cross_validate(n_folds, data_dir, categories, image_size, num_epochs):
	# initialize stratifying k fold
	skf = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = SEED)

	# open pickle files
	# features = pickle.load(open("features.pickle","rb")) #already reshaped as numpy array
	# features = features/255.0 
	# labels = pickle.load(open("labels.pickle","rb"))
	# labels = np.array(labels)
	# img_names = pickle.load(open("img_names.pickle","rb"))

	# data frame to save values of loss and validation after each fold
	df = pd.DataFrame()
	#obtain images
	data = import_images(data_dir, categories, image_size)
	features = data[0]
	labels = data[1]
	print("Stored features and labels")
	# for roc plotting
	tprs = []
	aucs = []
	mean_fpr = np.linspace(0, 1, 100)

	for index, (train_indices, val_indices) in enumerate(skf.split(features, labels)):
		print("Training on fold " + str(index + 1) + "/" + str(n_folds))
		train_features = features[train_indices]
		train_labels = labels[train_indices]
		print("Training data obtained")
		val_features = features[val_indices]
		val_labels = labels[val_indices]
		print("Validation data obtained")
		# train_labels, val_labels = labels[train_indices], labels[val_indices]

		# Create new model each time
		model = None
		model = build_model(image_size)
		print("Training model")
		es_callback = EarlyStopping(monitor = 'val_loss', patience = 4, restore_best_weights = True)
		history = model.fit(train_features, train_labels, batch_size=64, epochs = num_epochs, callbacks = [es_callback], validation_data = (val_features, val_labels))
		# save values of loss and accuracy into df
		df = df.append([[index+1, history.history['loss'][num_epochs-1], history.history['acc'][num_epochs-1], history.history['val_loss'][num_epochs-1], history.history['val_acc'][num_epochs-1]]])
		# model_json = model.to_json()
		# with open("model.json", "w") as json_file :
		# 	json_file.write(model_json)

		# model.save_weights("model.h5")
		# print("Saved model to disk")

		model.save('saved_models/CNN_' + str(index + 1) + '.model')

		# Printing a graph showing the accuracy changes during the training phase
		print(history.history.keys())
		plt.figure(1)
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper left')
		plt.savefig('graphs/val_accuracy_' + str(index+1) + '.png')
		# plt.show()
		plt.clf()

		plt.figure(2)
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper left')
		plt.savefig('graphs/val_loss_' + str(index+1) + '.png')
		# plt.show()
		plt.clf()

		# roc curve stuff
		probas_ = model.predict_proba(val_features)
		# Compute ROC curve and area the curve
		fpr, tpr, thresholds = roc_curve(val_labels, probas_[:, 1])
		tprs.append(interp(mean_fpr, fpr, tpr))
		tprs[-1][0] = 0.0
		roc_auc = auc(fpr, tpr)
		aucs.append(roc_auc)
		# Plots ROC for each individual fold:
		# plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (index + 1, roc_auc))
		# use the mean statistics to compare each model (that we train/test using 10-fold cv)
		mean_tpr = np.mean(tprs, axis=0)
		mean_tpr[-1] = 1.0
		mean_auc = auc(mean_fpr, mean_tpr)
		std_auc = np.std(aucs)

		# plot the mean ROC curve and display AUC (mean/st dev)
		plotROCforKfold(mean_fpr, mean_tpr, mean_auc, std_auc)
	df = df.rename({0: 'Fold Number',\
					1: 'Training Loss',\
					2: 'Training Accuracy',\
					3: 'Validation Loss', \
					4: 'Validation Accuracy'}, axis='columns')
	df.to_csv(os.path.join('graphs','final_acc_loss.csv'), encoding='utf-8', index=False)
	
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser('import pickle files')
	# parser.add_argument('-p', '--pickle_dir', default='data_pickles', help='Folder for pickle files')
	parser.add_argument('-d', '--directory', default='', help='Folder holding category folders')	
	parser.add_argument('-c1', '--category1', default='lyco_train', help='Folder of class 1')
	parser.add_argument('-c2', '--category2', default='sela_train', help='Folder of class 2')
	parser.add_argument('-s', '--img_size', default=256, help='Image dimension in pixels')
	parser.add_argument('-n', '--number_folds', default=10, help='Number of folds for cross validation')
	parser.add_argument('-e', '--number_epochs', default=25, help='Number of epochs')

	args = parser.parse_args()
	# divided_data = groups_to_arrays(args.pickle_dir, args.number_groups)
	#model = build_model(args.img_size)
	categories = [args.category1, args.category2]
	
	if not os.path.exists('graphs'):
		os.makedirs('graphs')
	if not os.path.exists('saved_models'):
		os.makedirs('saved_models')
	
	train_cross_validate(int(args.number_folds), args.directory, categories, int(args.img_size), int(args.number_epochs))

