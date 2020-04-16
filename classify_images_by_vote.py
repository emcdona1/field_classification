import os
import argparse
import pandas as pd
import tensorflow as tf
from utilities.timer import Timer
import numpy as np
from datetime import datetime
from labeled_images.labeledimages import LabeledImages

THRESHOLD = 0.5


def main():
    # Start execution and parse arguments
    timer = Timer('Classifying a test set')
    image_folders, class_labels, model_directory = process_input_arguments()

    # Import images
    # TODO: Update this to use the create_models method, and to accept color mode as a param
    images = LabeledImages(1)
    images.load_images_from_folders(image_folders, 3, class_labels)
    # image_features, actual_class, img_filenames = import_images(img_directory, folders, img_size)
    print('Images imported.')

    combined_results = pd.DataFrame()

    for model_name in os.listdir(model_directory):
        model_path = os.path.join(model_directory, model_name)
        if os.path.isfile(model_path):
            # Load model
            model = tf.keras.models.load_model(model_path)
            print('Model ' + model_name + ' loaded.')

            # Generate predictions and organize results
            predictions = make_predictions(images.features, images.labels, images.img_names, images.class_labels, model)
            print('Predictions generated.')

            # if this is the first model being processed, add a column of the file names & actual classification
            if len(combined_results) == 0:
                combined_results['filename'] = predictions['filename']
                combined_results['actual_class'] = predictions['actual_class']
            # add newest predictions to results
            combined_results[model_name] = predictions[class_labels[1] + '_pred']  # probability of class = 1

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

    if not os.path.exists('predictions'):
        os.makedirs('predictions')
    write_dataframe_to_csv('predictions', 'model_vote_predict', combined_results)

    # TODO: for each row in chart, vote (simple majority) and give each *image* a final classification
    # TODO: output: image name, final classification, true classification, tp/fn/fp/tn, then each classification

    # Finish execution
    timer.stop()
    timer.print_results()


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
    prediction_class = prediction_integer_func(predictions[:, [1]])  # 0/1 labels of predictions

    prediction_label_func = np.vectorize(lambda t: class_labels[t])
    pred_actual_class_labels = np.c_[prediction_label_func(prediction_class), prediction_label_func(actual_class)]

    # Join all information into one nparray -> pd.DataFrame
    headers = ['filename', class_labels[0] + '_pred', class_labels[1] + '_pred', 'pred_class', 'actual_class',
               'pred_label', 'actual_label']
    joined_arrays = np.c_[
        img_filenames, predictions, prediction_class, actual_class, pred_actual_class_labels]
    predictions_to_write = pd.DataFrame(joined_arrays, columns=headers)

    return predictions_to_write


def write_dataframe_to_csv(folder, filename, dataframe_to_write):
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
    dataframe_to_write.to_csv(filepath, encoding='utf-8', index=False)

    return filepath


def process_input_arguments():
    parser = argparse.ArgumentParser('Import a model and classify images.')
    parser.add_argument('c1', help='Path of class 1 images')
    parser.add_argument('c2', help='Path of class 2 images')
    parser.add_argument('models', help='Folder of models to use')
    # TODO: add color mode arg
    args = parser.parse_args()

    image_folders = (args.c1, args.c2)
    class_labels = parse_class_names_from_image_folders(args)
    model_directory = args.models

    return image_folders, class_labels, model_directory


def parse_class_names_from_image_folders(args) -> tuple:
    class1 = args.c1.strip(os.path.sep)
    class2 = args.c2.strip(os.path.sep)
    class1 = class1.split(os.path.sep)[class1.count(os.path.sep)]
    class2 = class2.split(os.path.sep)[class2.count(os.path.sep)]
    return class1, class2


if __name__ == '__main__':
    main()
