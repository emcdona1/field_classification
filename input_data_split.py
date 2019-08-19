# splits training and testing data into separate pickle files
import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
import pickle
import math
import argparse

def split_data(all_data, data_dir, categories): 
    for category in categories:
        path=os.path.join(data_dir,category) #look at each folder of images
        class_index = categories.index(category)
        for img in os.listdir(path): # look at each image
            try:
                img_array = cv2.imread(os.path.join(path,img), -1) #-1 means image is read as color
                all_data.append([img_array, class_index,img])
            except Exception as e:
                pass
    random.shuffle(all_data)
    print("Loaded and shuffled data")
    return all_data

#use pickle to export the data
def store_data(data_dir, all_data, percent_test,image_size): #percent_test is how much of the data goes to testing data
    if (percent_test > 1):
        print("Invalid percent")
        return
    features = []
    labels = []
    img_names = []
    #store the image features (array of RGB for each pixel) and labels into corresponding arrays
    for data_feature, data_label, file_name in all_data:
        features.append(data_feature)
        labels.append(data_label)
        img_names.append(file_name)


    #reshape into array
    features = np.array(features).reshape(-1, image_size, image_size, 3) #3 bc three channels for RGB values
    num_training = math.floor((1.0-percent_test)*len(features))

    #use pickle for image features
    training_features = features[:num_training]
    test_features = features[num_training:]
    pickle_out = open(data_dir + "/features.pickle", "wb") #wb = write binary
    pickle.dump(training_features, pickle_out) #(output file, source)
    pickle_out.close()
    pickle_out = open(data_dir + "/test_features.pickle", "wb") #wb = write binary
    pickle.dump(test_features, pickle_out) #(output file, source)
    pickle_out.close()

    #use pickle for image labels
    training_labels = labels[:num_training]
    test_labels = labels[num_training:]
    pickle_out = open(data_dir + "/labels.pickle", "wb")
    pickle.dump(training_labels, pickle_out)
    pickle_out.close()
    pickle_out = open(data_dir + "/test_labels.pickle", "wb") #wb = write binary
    pickle.dump(test_labels, pickle_out) #(output file, source)
    pickle_out.close()

    training_names = img_names[:num_training]
    test_names = img_names[num_training:]
    pickle_out = open(data_dir + "/img_names.pickle","wb")
    pickle.dump(training_names, pickle_out)
    pickle_out.close()
    pickle_out = open(data_dir + "/test_img_names.pickle","wb")
    pickle.dump(test_names, pickle_out)
    pickle_out.close()
    print("Data stored in pickle files")

if __name__== '__main__':
    parser = argparse.ArgumentParser('data to be imported')
    parser.add_argument('-d', '--directory', default='frullania', help='Folder holding category folders')
    parser.add_argument('-c1', '--category1', default='coastal_rsz', help='1st folder')
    parser.add_argument('-c2', '--category2', default='rostrata_rsz', help='2nd folder')
    parser.add_argument('-s', '--image_size', default = '256', help="Image size")
    parser.add_argument('-p', '--percent_test', default =0.1, help='percentage between [0,1] to go to testing')
    args = parser.parse_args()
    categories = [args.category1, args.category2]
    all_data= []
    all_data = split_data(all_data, args.directory, categories)
    store_data(args.directory, all_data, args.percent_test, int(args.image_size))