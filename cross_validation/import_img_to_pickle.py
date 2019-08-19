# --- CONSTANTS ---
SEED = 1

# Import statements
import argparse
import numpy as np
import os
import random
random.seed(a=SEED)
import cv2
import pickle
from sklearn.model_selection import StratifiedKFold
from numpy.random import seed

def import_images(data_dir, categories, image_size): 
    all_data = []
    for category in categories:
        path=os.path.join(data_dir,category) #look at each folder of images
        class_index = categories.index(category)
        for img in os.listdir(path): # look at each image
            try:
                img_array = cv2.imread(os.path.join(path,img), -1) #-1 means image is read as color
                all_data.append([img_array, class_index, img])
            except Exception as e:
                pass
    random.shuffle(all_data)
    print("Loaded and shuffled data")

    features = []
    labels = []
    img_names = []

    #store the image features (array of RGB for each pixel) and labels into corresponding arrays
    for data_feature, data_label, file_name in all_data:
        features.append(data_feature)
        labels.append(data_label)
        img_names.append(file_name)


    #reshape into numpy array
    features = np.array(features).reshape(-1, image_size, image_size, 3) #3 bc three channels for RGB values
    labels = np.array(labels)

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
    

if __name__== '__main__':
    parser = argparse.ArgumentParser('data to be imported')
    parser.add_argument('-d', '--directory', default='', help='Folder holding category folders')
    parser.add_argument('-p', '--pickle_dir', default='data_pickles', help='Folder for pickle files')
    parser.add_argument('-c1', '--category1', default='Lycopodiaceae_cv', help='Folder of class 1')
    parser.add_argument('-c2', '--category2', default='Selaginellaceae_cv', help='Folder of class 2')
    parser.add_argument('-s', '--image_size', default = '256', help="Image size")
    parser.add_argument('-n', '--num_groups', default = '10', help="Number of groups for cross validation")
    args = parser.parse_args()
    categories = [args.category1, args.category2]
    import_images(args.directory, categories, int(args.image_size))
