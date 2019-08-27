# from keract import get_activations #pip install keract
# from keract import display_activations

# import cv2
# import tensorflow as tf
# import numpy as np
# import pandas as pd
# import os
# import random
# import pickle

# model = tf.keras.models.load_model("saved_models/8.22.19_12.20_model/CNN_1.model") 
# img_arr = cv2.imread('lyco_holdoff/C0265057F_68132_rsz.jpg')
# img_arr = np.array(img_arr).reshape(-1, 128, 128,3)    
# img_arr = img_arr/255.0

# activations = get_activations(model, img_arr)
# display_activations(activations)

from keras.models import Model
import tensorflow as tf
model = tf.keras.models.load_model("saved_models/8.22.19_12.20_model/CNN_1.model") 
model.summary()
for layer in model.layers:
    if 'conv' not in layer.name:
        continue
    filters,biases = layer.get_weights()
    print(layer.name, filters.shape)
layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs = model.input, outputs = layer_outputs)