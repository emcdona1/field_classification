import os
import argparse
import time
from datetime import datetime
import classify_images
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

THRESHOLD = (1 - 0.7395)

def process_input_arguments():
    parser = argparse.ArgumentParser('Import a model and classify images.')
    parser.add_argument('-d', '--directory', default='images', help='Folder containing the image folders')	
    parser.add_argument('-c1', '--category1', help='Folder of class 1 images')
    parser.add_argument('-c2', '--category2', help='Folder of class 2 images')
    parser.add_argument('-s', '--img_size', default=256, help='Image dimension in pixels')
    parser.add_argument('-m', '--models', help='Folder of models to use')
    args = parser.parse_args()

    img_directory = args.directory
    folders = [args.category1, args.category2]
    img_size = args.img_size
    model_directory = args.models

    return img_directory, folders, img_size, model_directory

if __name__ == '__main__':
    # Start execution and parse arguments
    start_time = time.time()
    img_directory, folders, img_size, model_directory = process_input_arguments()
    # Map class numbers to class labels
    class_labels = [ folders[0].split('_')[0], folders[1].split('_')[0] ]

    # Import images
    pixel_values, actual_class, img_filenames = classify_images.import_images(img_directory, folders, img_size)
    print('Images imported.')

    combined_results = pd.DataFrame()

    for model_name in os.listdir(model_directory):
        model_path = os.path.join(model_directory, model_name)
        if os.path.isfile(model_path):
            # Load model
            model = tf.keras.models.load_model(model_path)
            print('Model ' + model_name + ' loaded.')

            # Generate predictions and organize results
            predictions = classify_images.make_predictions(pixel_values, actual_class, img_filenames, class_labels, model)
            print('Predictions generated.')

            # if this is the first model being processed, add a column of the file names & actual classification
            if len(combined_results) == 0:
                combined_results['filename'] = predictions['filename']
                combined_results['actual_class'] = predictions['actual_class']
            # add newest predictions to results
            combined_results[model_name] = predictions[class_labels[1] + '_pred'] # probability of class = 1

    combined_results['label'] = -1
    combined_results['tp'] = 0
    combined_results['fn'] = 0
    combined_results['fp'] = 0
    combined_results['tn'] = 0
    for (idx, row) in combined_results.iterrows():
        count = 0
        for p in range(2, 7):
            if float(row[p]) > THRESHOLD:
                count = count + 1
        prediction = 1 if count >= 3 else 0
        combined_results.at[idx, 'label'] = prediction
        actual = int(row['actual_class'])
        if actual == 1:
            if prediction == 1:
                combined_results.at[idx, 'tp'] = 1
            elif prediction == 0:
                combined_results.at[idx, 'fn'] = 1
            else:
                print('Invalid prediction value')
        elif actual == 0:
            if prediction == 1:
                combined_results.at[idx, 'fp'] = 1
            elif prediction == 0:
                combined_results.at[idx, 'tn'] = 1
            else:
                print('Invalid prediction value')
        else:
            print('Invalid image class value')



    classify_images.write_dataframe_to_CSV('predictions','model_vote_predict', combined_results)

    # TODO: for each row in chart, vote (simple majority) and give each *image* a final classification
    # TODO: output: image name, final classification, true classification, tp/fn/fp/tn, then each classification

    # Finish execution
    end_time = time.time()
    print('Completed in %.1f seconds' % (end_time - start_time))