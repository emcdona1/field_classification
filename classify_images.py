import os
import argparse
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

def import_images(img_directory, folders):
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
    features = features.reshape(-1, 256, 256, 3) # 3 bc three channels for RGB values
        # -1 means "numpy figure out this dimension," so the new nparray has the dimensions of: [#_of_images rows, img_size, img_size, 3] 
    labels = np.array(labels)
    return features, labels, img_names

def confusion_matrix(prediction_class_labels, actual_class_labels):
    # columns: tp, fn, fp, tn
    conf_mat = np.zeros((len(prediction_class_labels), 4))
    
    for idx, pred in enumerate(prediction_class_labels):
        if pred == 1:
            if pred == actual_class_labels[idx]:
                conf_mat[idx][0] = 1 # true positive
            else:
                conf_mat[idx][2] = 1 # false positive
        elif pred == 0:
            if pred == actual_class_labels[idx]:
                conf_mat[idx][3] = 1 # true negative
            else:
                conf_mat[idx][1] = 1 # false negative
        else:
            print('Invalid value for prediction class!')
         
    return conf_mat

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Import a model and classify images.')
    parser.add_argument('-d', '--directory', default='images', help='Folder holding category folders')	
    parser.add_argument('-c1', '--category1', help='Folder of class 1')
    parser.add_argument('-c2', '--category2', help='Folder of class 2')
    parser.add_argument('-m', '--model', help='Filepath of model to use')
    args = parser.parse_args()
    img_directory = args.directory
    folders = [args.category1, args.category2]
    model_directory = args.model

    # Load model
    model = tf.keras.models.load_model(model_directory)

    # Import images: returns
    pics, actual_class, img_names = import_images(img_directory, folders)
    # pics = pixels, actual_class = 0/1 labels of actual classification, img_names = file names
    
    # Predict classes of imported images
    predictions = model.predict(pics)
    pfunc = np.vectorize(lambda t: int(round(t)))
    prediction_class = pfunc(predictions[:,[1]]) # 0/1 labels of predictions

    # Map class numbers to class labels
    maps = [ folders[0].split('_')[0], folders[1].split('_')[0] ]
    mfunc = np.vectorize(lambda t: maps[t])
    prediction_class_labels = mfunc(prediction_class) # class name labels of predictions
    actual_class_labels = mfunc(actual_class) # class name labels of actual classification

    # Calculate confusion matrix: tp, fn, fp, tn
    conf_matrix = confusion_matrix(prediction_class, actual_class)

    # Join all information into one nparray -> pd.DataFrame
    headers =           ['filename', maps[0] + '_pred',  maps[1] + '_pred', 'class_pred',            'actual',      'actual_class',         'tp', 'fn', 'fp', 'tn']
    pred_joined = np.c_ [img_names,  predictions,                           prediction_class_labels, actual_class,  actual_class_labels, conf_matrix]
    predictions_final = pd.DataFrame(pred_joined, columns=headers)

    # save to file
    timestamp = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
    predictions_final.to_csv(os.path.join('predictions',timestamp+'predictions.csv'), encoding='utf-8',index=False)