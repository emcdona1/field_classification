import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import random
import pickle

# CATEGORIES = ['Lycopodiaceae', 'Selaginellaceae']
CATEGORIES = ['lyco_sample_test', 'sela_sample_test']
IMG_SIZE = 256
root = 'data'

test_features = pickle.load(open("test_features.pickle","rb"))
test_features = test_features/255.0
test_labels = pickle.load(open("test_labels.pickle","rb"))
test_names = pickle.load(open("test_img_names.pickle","rb"))

test_data = []
df = pd.DataFrame()
model = tf.keras.models.load_model("CNN.model") #, custom_objects={'custom_activation':Activation(custom_activation)})

for i in range(len(test_features)):
    img_arr = np.array(test_features[i]).reshape(1, IMG_SIZE, IMG_SIZE,3)    
    try:
        prediction = model.predict(img_arr)
        prediction = list(prediction[0])
        final_prediction = CATEGORIES[prediction.index(max(prediction))]
        df = df.append([[test_names[i],CATEGORIES[test_labels[i]],final_prediction]])
    except Exception as e:
            pass
df = df.rename({0:'Image Name', 1: 'True Label', 2:'Predicted Label'}, axis='columns')
df.to_csv(root+'/test_results.csv', encoding= 'utf-8', index=False)
print("done")
