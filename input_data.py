import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
import pickle

DATA_DIR = 'data'
# CATEGORIES = ['Lycopodiaceae', 'Selaginellaceae']
CATEGORIES = ['lyco_test_hundred', 'sela_test_hundred']
IMG_SIZE = 256 #pixels

#create an array that holds two items: arrays for each pixel in each image and the label for that image
def create_training_data(training_data):
    for category in CATEGORIES:
        path=os.path.join(DATA_DIR,category) #look at each folder of images
        class_index = CATEGORIES.index(category)
        for img in os.listdir(path): # look at each image
            try:
                img_array = cv2.imread(os.path.join(path,img), -1) #-1 means image is read as color
                training_data.append([img_array, class_index])
            except Exception as e:
                pass
    random.shuffle(training_data)
    return training_data

#use pickle to export the data
def store_training_data(training_data):
    features = []
    labels = []
    #store the image features (array of RGB for each pixel) and labels into corresponding arrays
    for data_feature, data_label in training_data:
        features.append(data_feature)
        labels.append(data_label)

    #reshape into array
    features = np.array(features).reshape(-1, IMG_SIZE, IMG_SIZE, 3) #3 bc three channels for RGB values

    #use pickle for image features
    pickle_out = open("features.pickle", "wb") #wb = write binary
    pickle.dump(features, pickle_out) #(output file, source)
    pickle_out.close()

    #use pickle for image labels
    pickle_out = open("labels.pickle", "wb")
    pickle.dump(labels, pickle_out)
    pickle_out.close()


training_data= []
training_data = create_training_data(training_data)
store_training_data(training_data)

# just checking that we all good
pickle_in = open("features.pickle", "rb")
X = pickle.load(pickle_in)
print('hi')