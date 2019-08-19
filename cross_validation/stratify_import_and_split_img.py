import argparse
import numpy as np
import os
import random
random.seed(a=1)
import cv2
import pickle
from sklearn.model_selection import StratifiedKFold
from numpy.random import seed


# def import_images(all_data, data_dir, categories): 
#     all_data = []
#     for category in categories:
#         path=os.path.join(data_dir,category) #look at each folder of images
#         class_index = categories.index(category)
#         categ_arr = []
#         for img in os.listdir(path): # look at each image
#             try:
#                 img_array = cv2.imread(os.path.join(path,img), -1) #-1 means image is read as color
#                 categ_arr.append([img_array, class_index, img])
#             except Exception as e:
#                 pass
#         random.shuffle(categ_arr)
#         all_data.append(categ_arr)
#     print("Loaded and shuffled data")
#     return all_data

def import_images(all_data, data_dir, categories): 
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
    return all_data

def store_data(pickle_data_dir, all_data, num_groups, image_size): #percent_test is how much of the data goes to testing data
    #store the image features (array of RGB for each pixel) and labels into corresponding arrays
    # for data_feature, data_label, file_name in all_data:
    #     features.append(data_feature)
    #     labels.append(data_label)
    #     img_names.append(file_name)

    #reshape into array
    # features = np.array(features).reshape(-1, image_size, image_size, 3) #3 bc three channels for RGB values
    num_categories = len(all_data)
    # Calculate number of images per group
    categ_total_images = []
    APPROX_NUM_PER_GROUP = [] #this one's approximate (may be off by one due to imperfect divisions)
    remainder = []
    indices = [] # "global"; never reset
    group_img_counter = []
    num_images_in_group = []
    group_full = []
    for i in range(num_categories):
        categ_total_images.append(len(all_data[i]))
        APPROX_NUM_PER_GROUP.append(total_images[i]/num_groups)
        remainder.append(total_images[i]%num_groups)
        indices.append(0)
        group_img_counter.append(0)
        num_images_in_group.append(0)
        group_full.append(False)

    features = []
    labels = []
    img_names = []
    total_images_in_group = 0
    for j in range(num_groups):
        ctr = 0 # keep track of how many images we have in each group
        # Set number of images to be in ith group
        for i in range(num_categories):
            if j < remainder[i]:
                num_images_in_group[i] = APPROX_NUM_PER_GROUP + 1
            else:
                num_images_in_group[i] = APPROX_NUM_PER_GROUP
            # reset counter for number of each category in this group
            group_img_counter[i] = 0
            total_images_in_group +=num_images_in_group[i]
        group = []
        group_features = []
        group_labels = []
        group_names = []
        total_group_ctr = 0
        while total_group_ctr < total_images_in_group:
            # randomly choose a group to pick an image from
            pick = random.randint(0,1)
            # Show image for test (some reason it's not responding)
            # cv2.imshow(all_data[all_data_index][2], all_data[all_data_index][0])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            group_features.append(all_data[all_data_index][0])
            group_labels.append(all_data[all_data_index][1])
            group_names.append(all_data[all_data_index][2])
            all_data_index += 1 # never reset
            ctr += 1 # gets reset at each group  
        group_features = np.array(group_features).reshape(-1, image_size, image_size, 3) # 3 bc three channels for RGB values
        # Export data as pickle files
        pickle_out = open(pickle_data_dir + "/" + str(j) + "_features.pickle", "wb") #wb = write binary
        pickle.dump(group_features, pickle_out) #(output file, source)
        pickle_out.close()

        pickle_out = open(pickle_data_dir + "/" + str(j) + "_labels.pickle", "wb") #wb = write binary
        pickle.dump(group_labels, pickle_out) #(output file, source)
        pickle_out.close()

        pickle_out = open(pickle_data_dir + "/" + str(j) + "_names.pickle","wb")
        pickle.dump(group_names, pickle_out)
        pickle_out.close()
        
        # pickle_in = open(pickle_data_dir+"/0_features.pickle","rb")
        # x = pickle.load(pickle_in)
    print("done")

def to_pickle(num_groups, all_data, image_size):
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
    all_data= []
    all_data = import_images(all_data, args.directory, categories)
    to_pickle(2, all_data, int(args.image_size))
    #store_data(args.pickle_dir, all_data, int(args.num_groups), int(args.image_size))