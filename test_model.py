import cv2
import tensorflow as tf
import numpy as np

# CATEGORIES = ['Lycopodiaceae', 'Selaginellaceae']
CATEGORIES = ['lyco_test_hundred', 'sela_test_hundred']
IMG_SIZE = 256

model = tf.keras.models.load_model("CNN.model")
image = "C0611522F_23738_rsz.jpg"
img_array = cv2.imread(image, -1) #-1 means image is read as color
img_array = np.array(img_array).reshape(-1, IMG_SIZE, IMG_SIZE, 3) #3 bc three channels for RGB values
prediction = model.predict([img_array])
prediction = list(prediction[0])
print(CATEGORIES[prediction.index(max(prediction))])