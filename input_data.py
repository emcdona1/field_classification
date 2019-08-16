import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
import pickle
import math
import argparse

DATA_DIR = 'data'
# CATEGORIES = ['Lycopodiaceae', 'Selaginellaceae']
# CATEGORIES = ['lyco_train', 'sela_train']
IMG_SIZE = 256 #pixels

# create an array that holds two items: arrays for each pixel in each image and the label for that image
def split_data(categories):
    all_data = []
    for category in categories:
        path=os.path.join(DATA_DIR,category) #look at each folder of images
        class_index = categories.index(category)
        for img in os.listdir(path): # look at each image
            try:
                img_array = cv2.imread(os.path.join(path,img), -1) #-1 means image is read as color
                training_data.append([img_array, class_index,img])
            except Exception as e:
                pass
    random.shuffle(all_data)
    return all_data

#use pickle to export the data
def store_training_data(training_data): 
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
    pickle_out = open("features.pickle", "wb") #wb = write binary
    pickle.dump(features, pickle_out) #(output file, source)
    pickle_out.close()

    #use pickle for image labels
    pickle_out = open("labels.pickle", "wb")
    pickle.dump(labels, pickle_out)
    pickle_out.close()

    pickle_out = open("img_names.pickle","wb")
    pickle.dump(img_names, pickle_out)
    pickle_out.close()
    print("Data stored in pickle files")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('images to import')
    parser.add_argument('-c1', '--category_1', default='lyco_train', help='folder where images of class A are')
    parser.add_argument('-c2', '--category-2', default='sela_train', help='folder where images of class B are')
    args = parser.parse_args()
    categories = [args.category_1, args.category_2]
    
