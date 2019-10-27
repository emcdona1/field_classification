import os
import re
import argparse
import time
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

THRESHOLD = (1 - 0.7395)

def process_input_arguments():
    parser = argparse.ArgumentParser('Import a model and classify images.')
    parser.add_argument('-d', '--directory', default='images', help='Folder holding category folders')	
    parser.add_argument('-c1', '--category1', help='Folder of class 1')
    parser.add_argument('-c2', '--category2', help='Folder of class 2')
    parser.add_argument('-s', '--img_size', default=256, help='Image dimension in pixels')
    parser.add_argument('-m', '--model', help='Filepath of model to use')
    args = parser.parse_args()

    img_directory = args.directory
    folders = [args.category1, args.category2]
    img_size = args.img_size
    model_directory = args.model

    return img_directory, folders, img_size, model_directory

def import_images(img_directory, folders, img_size):
    ''' Imports images from 2 folders into program

    Parameters:
    -----
    img_directory : directory of images folders
    folders : list of two strings; folders[0] is the folder of images with classification = 0, folders[1] is classification 1

    Output:
    -----
    features : nparray (shape = #images x img_size x img_size x 3) of ints, containing the RGB pixel values for each image, i.e. the inputs for the model
    labels : nparray (shape = #images x 1) of ints, containing actual classification (based on the image folder)
    img_names : nparray (shape = #images x 1) of strings, contains all filenames
    '''
    all_data = []
    for category in folders:
        path=os.path.join(img_directory,category) #look at each folder of images
        class_index = folders.index(category)
        for img in os.listdir(path): # look at each image
            try:
                img_array = cv2.imread(os.path.join(path,img), -1) #-1 means image is read as color
                img_array = img_array/255.0
                all_data.append([img_array, class_index,img]) #, img])
            except Exception as e:
                pass
    features = []
    labels = []
    img_names = []

	#store the image features (array of RGB for each pixel) and labels into corresponding arrays
    for data_feature, data_label, img in all_data:
        features.append(data_feature)
        labels.append(data_label)
        img_names.append(img)

    #reshape into numpy array
    features = np.array(features) #turns list into a numpy array
    features = features.reshape(-1, img_size, img_size, 3) # 3 bc three channels for RGB values
        # -1 means "numpy figure out this dimension," so the new nparray has the dimensions of: [#_of_images rows, img_size, img_size, 3] 
    labels = np.array(labels)
    return features, labels, img_names

def make_predictions(pixel_values, actual_class, img_filenames, class_labels, model):
    ''' Model predicts classifications for all images, and organizes into a DataFrame

    Parameters:
    -----
    @pixel_values : numpy array of RBG pixel values for each image
    @actual_class : numpy array of 0/1's of actual classification of the images
    @img_filenames : numpy array of the image filenames
    @class_labels : list containing the names of the two classes (e.g. ['coastal', 'rostrata'])
    @model : keras model, already loaded

    Output:
    -----
    DataFrame with the following columns:
    1. image filename (string)
    2. prediction of class = 0 (float)
    3. prediction of class = 1 (float)
    4. class predition - argmax (int, 0 or 1)
    5. actual class (int, 0 or 1)
    6. predicted class label (string)
    7. actual class label (string)
    8. True Positive (1 if the image was correctly predicted to be class=1, 0 otherwise)
    9. False Negative (1 if the image was incorrectly predicted to be class=1, 0 otherwise)
    10. False Positive (1 if the image was incorrectly predicted to be class=0, 0 otherwise)
    11. True Negative (1 if the image was correctly predicted to be class=0, 0 otherwise)
    '''
    # Predict classes of imported images
    predictions = model.predict(pixel_values)
    prediction_integer_func = np.vectorize(lambda t: (1 if t > THRESHOLD else 0))
    prediction_class = prediction_integer_func(predictions[:,[1]]) # 0/1 labels of predictions

    prediction_label_func = np.vectorize(lambda t: class_labels[t])
    pred_actual_class_labels = np.c_[prediction_label_func(prediction_class), prediction_label_func(actual_class)]
    
    # Calculate confusion matrix: tp, fn, fp, tn
    conf_matrix = confusion_matrix(prediction_class, actual_class)

    # Join all information into one nparray -> pd.DataFrame
    headers = ['filename', class_labels[0] + '_pred', class_labels[1] + '_pred', 'pred_class', 'actual_class', 'pred_label', 'actual_label', 'tp', 'fn', 'fp', 'tn']
    joined_arrays = np.c_ [img_filenames, predictions, prediction_class, actual_class, pred_actual_class_labels, conf_matrix]
    predictions_to_write = pd.DataFrame(joined_arrays, columns=headers)

    return predictions_to_write

def confusion_matrix(prediction_class_labels, actual_class_labels):
    ''' Determines confusion matrix value for each tuple.

    PARAMATERS:
    -----
    prediction_class_labels: numpy array with 0/1 predicted classifications (shape = # of images x 1)
    actual_class_labels: numpy array with 0/1 actual classifications (shape = # of images x 1)

    OUTPUT:
    -----
    numpy array (shape = # of images x 4), where:
    - col 1 = true positive
    - col 2 = false negative
    - col 3 = false positives
    - col 4 = true negatives
    '''
    # columns: 0=tp, 1=fn, 2=fp, 3=tn
    conf_mat = np.zeros((len(prediction_class_labels), 4))
    
    for idx, pred in enumerate(prediction_class_labels):
        actual = actual_class_labels[idx]
        if pred == 1:
            if pred == actual:
                conf_mat[idx][0] = 1 # true positive
            else:
                conf_mat[idx][2] = 1 # false positive
        elif pred == 0:
            if pred == actual:
                conf_mat[idx][3] = 1 # true negative
            else:
                conf_mat[idx][1] = 1 # false negative
        else:
            print('Invalid value for prediction class!')
         
    return conf_mat

def write_dataframe_to_CSV(folder, filename, dataframe_to_write):
    ''' Writes the given DataFrame to a file.
    Parameters:
    -----
    @folder : String to designate folder in which to write file
    @filename : String to add designation to filename -- file names are timestamp+filename
    @dataframe_to_write : DataFrame to be written to CSV

    Output:
    -----
    File path of the written file
    '''
    timestamp = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
    filename = timestamp + filename + '.csv'
    filepath = os.path.join(folder, filename)
    dataframe_to_write.to_csv(filepath, encoding = 'utf-8', index = False)

    return filepath

if __name__ == '__main__':
    # Start execution and parse arguments
    start_time = time.time()
    img_directory, folders, img_size, model_directory = process_input_arguments()

    # Load model
    model = tf.keras.models.load_model(model_directory)
    print('Model loaded.')

    # Import images
    pixel_values, actual_class, img_filenames = import_images(img_directory, folders, img_size)
    print('Images imported.')

    # Map class numbers to class labels
    class_labels = [ folders[0].split('_')[0], folders[1].split('_')[0] ]
    
    # Generate predictions and organize results
    predictions_to_write = make_predictions(pixel_values, actual_class, img_filenames, class_labels, model)
    print('Predictions generated.')

    # Save to file
    if not os.path.exists('predictions'):
        os.makedirs('predictions')
    model_name = re.split('[/\\\\]+', model_directory)[2].split('.')[0]
    filename = write_dataframe_to_CSV('predictions', 'predictions'+model_name, predictions_to_write)
    print('File written to \'%s\'.' % filename)

    # Finish execution
    end_time = time.time()
    print('Completed in %.1f seconds' % (end_time - start_time))