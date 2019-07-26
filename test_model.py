import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import random

# CATEGORIES = ['Lycopodiaceae', 'Selaginellaceae']
CATEGORIES = ['lyco_sample_test', 'sela_sample_test']
IMG_SIZE = 256
root = 'data'

test_data = []
df = pd.DataFrame()
model = tf.keras.models.load_model("CNN.model") #, custom_objects={'custom_activation':Activation(custom_activation)})
for category in CATEGORIES:
    path = os.path.join(root, category)
    for img in os.listdir(path):
            img_name = os.path.join(path,img)
            img_array = cv2.imread(img_name, -1) #-1 means image is read as color
            img_array = np.array(img_array).reshape(-1, IMG_SIZE, IMG_SIZE, 3) #3 bc three channels for RGB values
            img_array = img_array/255.0
            test_data.append([img,img_array,category])
random.shuffle(test_data)

for entry in test_data:
    try:
        prediction = model.predict([entry[1]])
        prediction = list(prediction[0])
        final_prediction = CATEGORIES[prediction.index(max(prediction))]
        df = df.append([[entry[0],entry[2],final_prediction]])
    except Exception as e:
            pass
df = df.rename({0:'Image Name', 1: 'True Label', 2:'Predicted Label'}, axis='columns')
df.to_csv(root+'/test_results.csv', encoding= 'utf-8', index=False)
print("done")

# random.shuffle(test_data)
# image = "C0611522F_23738_rsz.jpg"
# img_array = cv2.imread(image, -1) #-1 means image is read as color
# img_array = np.array(img_array).reshape(-1, IMG_SIZE, IMG_SIZE, 3) #3 bc three channels for RGB values
# prediction = model.predict([img_array])
# prediction = list(prediction[0])
# print(CATEGORIES[prediction.index(max(prediction))])