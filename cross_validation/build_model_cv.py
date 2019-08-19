# --- CONSTANTS ---
SEED = 1
# imports for reproducible results
from numpy.random import seed
seed(SEED)
from tensorflow.compat.v1 import set_random_seed
set_random_seed(SEED)

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


#from stratify_import_and_split_img import *

# returns a numpy array with all the data
# def groups_to_arrays(pickle_data_dir, num_groups):
# 	sets = []
# 	for i in range(num_groups):
# 		features = pickle.load(open(os.path.join(pickle_data_dir,str(i)+"_features.pickle"),"rb"))
# 		# normalize data
# 		features = features/255.0
# 		labels = pickle.load(open(os.path.join(pickle_data_dir,str(i)+"_labels.pickle"),"rb"))
# 		#names = pickle.load(open(os.path.join(pickle_data_dir,str(i)+"_names.pickle"),"rb"))

# 		one_group = [features, labels]
# 		sets.append(one_group)
# 	return sets

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
	model.add(Dropout(0.5))

	model.add(Dense(500, activation="linear", activity_regularizer=regularizers.l2(0.01), kernel_regularizer=regularizers.l2(0.05))) # kernel_regularizer=regularizers.l2(0.1)))
	model.add(Dense(500, activation="relu", activity_regularizer=regularizers.l2(0.01), kernel_regularizer=regularizers.l2(0.05))) 
	#model.add(Activation("relu"))

	model.add(Dropout(0.25))
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
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
	plt.savefig('graphs/mean_ROC.png')
    plt.show()

def train_cross_validate(n_folds):
	skf = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = SEED)
	features = pickle.load(open("features.pickle","rb"))
	labels = pickle.load(open("labels.pickle","rb"))
	img_names = pickle.load(open("img_names.pickle","rb"))

	# for roc plotting
	tprs = []
	aucs = []
	mean_fpr = np.linspace(0, 1, 100)

	for index, (train_indices, val_indices) in enumerate(skf.split(features, labels)):
		print("Training on fold " + str(index + 1) + "/10")
		train_features, val_features = features[train_indices], features[val_indices]
		train_labels, val_labels = labels[train_indices], labels[val_indices]

		# Create new model each time
		model = None
		model = build_model(256)

		history = model.fit(train_features, train_labels, batch_size=32, epochs = 3, validation_data = (val_features, val_labels))
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
		plt.savefig('graphs/val_accuracy_' + str(index) + '.png')
		# plt.show()

		plt.figure(2)
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper left')
		plt.savefig('graphs/val_loss' + str(index) + '.png')
		# plt.show()

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
	
		

if __name__ == '__main__':
	parser = argparse.ArgumentParser('import pickle files')
	parser.add_argument('-p', '--pickle_dir', default='data_pickles', help='Folder for pickle files')
	parser.add_argument('-n', '--number_groups', default=10, help='Number of groups for cross validation')
	parser.add_argument('-s', '--img_size', default=256, help='Image dimension in pixels')
	args = parser.parse_args()
	# divided_data = groups_to_arrays(args.pickle_dir, args.number_groups)
	#model = build_model(args.img_size)
	train_cross_validate(10)

# DATA_DIR = 'data'
# # CATEGORIES = ['Lycopodiaceae', 'Selaginellaceae']
# # CATEGORIES = ['lyco_train', 'sela_train']
# IMG_SIZE = 256 #pixels

# # use functions from input_data.py to shuffle data and store it
# training_data= []
# testing_data = []
# # training_data = split_data(training_data) #,testing_data)
# # store_training_data(training_data, 0.0)

# # Open up those pickle files
# features = pickle.load(open("features.pickle","rb"))
# labels = pickle.load(open("labels.pickle","rb"))

# # checking that images are labeled correctly
# # for i in range(5):
# #     test_img=features[i]
# #     cv2.imshow(img_names[i] + ' ' + str(labels[i]),test_img)
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()

# print("Files loaded")
# # build the model? let's give this a shot lol



# # Compiling the model using some basic parameters
# # learning rate start at 0.0001 in the Smithsonian paper as well
# opt = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=0.00001, decay=0.0, amsgrad=False)
# model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# print("Model created")
# # Training the model, with 10 iterations
# # validation_split corresponds to the percentage of images used for the validation phase compared to all the images
# # es_callback = EarlyStopping(monitor = 'val_loss', patience = 3, restore_best_weights = True)
# # history = model.fit(features, labels, batch_size=16, epochs=15, validation_split=0.22) #.22 of the .9 is .2 of the total
# # history = model.fit(features, labels, batch_size=32, epochs=15, callbacks=[es_callback], validation_split=0.22) #.22 of the .9 is .2 of the total
# history = model.fit(features, labels, batch_size=32, epochs = 30, validation_split=0.22) #.22 of the .9 is .2 of the total

# # Saving the model
