# test_model.py uses the images in the same folders in the training/validation folders. 
# This file allows us to test with other folders of images without retraining model
import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import random
import pickle
import argparse

IMG_SIZE = 256
root = 'data'

def test_model(model_folder_path, cat_root, categories, img_size):
    test_data = []
    df = pd.DataFrame()

    for category in categories:
        path = os.path.join(cat_root, category)
        for img in os.listdir(path):
                img_name = os.path.join(path,img)
                img_array = cv2.imread(img_name, -1) #-1 means image is read as color
                img_array = np.array(img_array).reshape(-1, img_size, img_size, 3) #3 bc three channels for RGB values
                img_array = img_array/255.0
                test_data.append([img,img_array,category])
    random.shuffle(test_data)
    print("Obtained all test images")
    models = []
    for model_name in os.listdir(model_folder_path):
        if not model_name[-6:] == '.model':
            continue
        models.append(tf.keras.models.load_model(os.path.join(model_folder_path,model_name)))
    print("Opened all models")
    for entry in test_data:
        predict_sums = [0,0]
        for model in models:
            try:
                prediction = model.predict(entry[1])
                prediction = list(prediction[0])
                predict_sums[0] += prediction[0]
                predict_sums[1] += prediction[1]
            except Exception as e:
                    pass
        final_prediction = categories[predict_sums.index(max(predict_sums))]
        df = df.append([[entry[0],entry[2],final_prediction]])

    df = df.rename({0:'Image Name', 1: 'True Label', 2:'Predicted Label'}, axis='columns')
    df.to_csv(os.path.join(cat_root,'test_results.csv'), encoding= 'utf-8', index=False)
    print("done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('test the model')
    parser.add_argument('-m', '--model_folder_path', default='saved_models', help='path to folder containing models named as such: CNN_[index].model. EX: CNN_1.model')
    parser.add_argument('-r', '--category_root', default='', help = 'go to folder holding category/image folders')
    parser.add_argument('-c1', '--category_1', default='adiantum_holdoff', help='first category')
    parser.add_argument('-c2', '--category_2', default='blechnum_holdoff', help='second category')
    parser.add_argument('-s', '--image_size', default=256, help='image size')
    args = parser.parse_args()
    categories = [args.category_1, args.category_2]
    test_model(args.model_folder_path, args.category_root, categories, int(args.image_size))
