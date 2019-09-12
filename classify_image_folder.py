import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import pandas as pd
import os
import datetime

def import_images():
    all_data = []
    folders = ['rostrata_micro', 'coastal_micro']
    img_directory = ''
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

if __name__ == '__main__':
    '''Currently this only works for hardcoding in the model used, and the frullania project.'''

    model = tf.keras.models.load_model('.\\saved_models\\CNN_1.model')
    pics, labels, img_names = import_images()
    predictions = model.predict(pics)
    maps = ['rostrata', 'coastal']
    mapper = lambda t: maps[int(round(t))]
    mfunc = np.vectorize(mapper)
    pred_class = mfunc(predictions[:,[1]])
    labels2 = mfunc(labels)

    headers = ['filename', 'rostrata_pred', 'coastal_pred', 'class_pred', 'actual', 'actual_class']
    pred_joined = np.c_[img_names, predictions, pred_class, labels, labels2]
    pred_final = pd.DataFrame(pred_joined, columns=headers)

    print(pred_final)
    timestamp = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S')
    pred_final.to_csv(os.path.join('predictions',timestamp+'predictions.csv'), encoding='utf-8',index=False)