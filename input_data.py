import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
import pickle
import math

DATA_DIR = 'data'
# CATEGORIES = ['Lycopodiaceae', 'Selaginellaceae']
CATEGORIES = ['Lycopodiaceae_selected', 'Selaginellaceae_selected']
IMG_SIZE = 256 #pixels

# create an array that holds two items: arrays for each pixel in each image and the label for that image
def split_data(training_data): #percent_test is how much of the data goes to testing data
    all_data = []
    for category in CATEGORIES:
        path=os.path.join(DATA_DIR,category) #look at each folder of images
        class_index = CATEGORIES.index(category)
        for img in os.listdir(path): # look at each image
            try:
                img_array = cv2.imread(os.path.join(path,img), -1) #-1 means image is read as color
                training_data.append([img_array, class_index,img])
            except Exception as e:
                pass
    random.shuffle(training_data)
    print("Images imported and data shuffled")
    return training_data

# def split_data(training_data, testing_data): 
#     for category in CATEGORIES:
#         path=os.path.join(DATA_DIR,category) #look at each folder of images
#         class_index = CATEGORIES.index(category)
#         for img in os.listdir(path): # look at each image
#             try:
#                 img_array = cv2.imread(os.path.join(path,img), -1) #-1 means image is read as color
#                 training_data.append([img_array, class_index,img])
#             except Exception as e:
#                 pass
#     random.shuffle(training_data)
#     print("Loaded and shuffled data")
#     return training_data

#use pickle to export the data
def store_training_data(training_data, percent_test): #percent_test is how much of the data goes to testing data
    # if (percent_test > 1):
    #     print("Invalid percent")
    #     return
    features = []
    labels = []
    img_names = []
    #store the image features (array of RGB for each pixel) and labels into corresponding arrays
    for data_feature, data_label, file_name in training_data:
        features.append(data_feature)
        labels.append(data_label)
        img_names.append(file_name)


    #reshape into array
    features = np.array(features).reshape(-1, IMG_SIZE, IMG_SIZE, 3) #3 bc three channels for RGB values
    num_training = math.floor((1.0-percent_test)*len(features))

    #use pickle for image features
    training_features = features[:num_training]
    # test_features = features[num_training:]
    pickle_out = open("features.pickle", "wb") #wb = write binary
    pickle.dump(training_features, pickle_out) #(output file, source)
    pickle_out.close()
    # pickle_out = open("test_features.pickle", "wb") #wb = write binary
    # pickle.dump(test_features, pickle_out) #(output file, source)
    # pickle_out.close()

    #use pickle for image labels
    training_labels = labels[:num_training]
    # test_labels = labels[num_training:]
    pickle_out = open("labels.pickle", "wb")
    pickle.dump(training_labels, pickle_out)
    pickle_out.close()
    # pickle_out = open("test_labels.pickle", "wb") #wb = write binary
    # pickle.dump(test_labels, pickle_out) #(output file, source)
    # pickle_out.close()

    training_names = img_names[:num_training]
    # test_names = img_names[num_training:]
    pickle_out = open("img_names.pickle","wb")
    pickle.dump(training_names, pickle_out)
    pickle_out.close()
    # pickle_out = open("test_img_names.pickle","wb")
    # pickle.dump(test_names, pickle_out)
    # pickle_out.close()
    print("Data stored in pickle files")


# training_data= []
# training_data = create_training_data(training_data)
# store_training_data(training_data)

# # just checking that we all good
# pickle_in = open("features.pickle", "rb")
# X = pickle.load(pickle_in)
# print('hi')