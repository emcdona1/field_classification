import os
import argparse
import pandas as pd
import tensorflow as tf
from utilities.timer import Timer
import numpy as np
from datetime import datetime
from labeled_images.labeledimages import LabeledImages
from cnnarguments import parse_class_names_from_image_folders
from labeled_images.colormode import ColorMode

THRESHOLD = 0.5


def main():
    # Start execution and parse arguments
    timer = Timer('Classifying a test set')
    image_folders, class_labels, list_of_models = process_input_arguments()

    # Import images
    images = LabeledImages()
    images.load_images_from_folders(image_folders, ColorMode.RGB, class_labels)
    print('Images imported.')

    combined_results = pd.DataFrame()
    combined_results['filename'] = images.image_names
    combined_results['actual_class'] = images.labels
    all_predictions = pd.DataFrame()

    for model_path in list_of_models:
        classify_images_with_a_model(class_labels, all_predictions, images, os.path.basename(model_path), model_path)

    all_predictions['voted_label'] = all_predictions.mean(axis=1)
    # calculate_confusion_matrix(combined_results)
    combined_results = combined_results.join(all_predictions)

    if not os.path.exists('predictions'):
        os.makedirs('predictions')
    write_dataframe_to_csv('predictions', 'model_vote_predict', combined_results)

    # TODO: for each row in chart, vote (simple majority) and give each *image* a final classification
    # TODO: output: image name, final classification, true classification, tp/fn/fp/tn, then each classification

    # Finish execution
    timer.stop()
    timer.print_results()


def calculate_confusion_matrix(combined_results):
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


def classify_images_with_a_model(class_labels: tuple, combined_results: pd.DataFrame,
                                 images: LabeledImages, model_name: str, model_path: str) -> None:
    if ".model" in model_path:
        # Load model
        model = tf.keras.models.load_model(model_path)
        print('Model ' + model_name + ' loaded.')

        # Generate predictions and organize results
        predictions = make_predictions(images, model)
        print('Predictions generated.')

        # add newest predictions to results
        combined_results[model_name] = predictions[class_labels[1] + '_pred']  # probability of class = 1
    else:
        print('Model file path error: "%s" is not a *.model file.' % model_path)


def make_predictions(images: LabeledImages, model: tf.keras.Model) -> pd.DataFrame:
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
    predictions: np.array = model.predict(images.features)
    prediction_integer_func = np.vectorize(lambda t: (1 if t > THRESHOLD else 0))
    prediction_class = prediction_integer_func(predictions[:, [1]])  # 0/1 labels of predictions

    prediction_label_func = np.vectorize(lambda t: images.class_labels[t])
    pred_actual_class_labels = np.c_[prediction_label_func(prediction_class), prediction_label_func(images.labels)]

    # Join all information into one nparray -> pd.DataFrame
    headers = [images.class_labels[0] + '_pred', images.class_labels[1] + '_pred']

    predictions_to_write = pd.DataFrame(predictions, columns=headers)

    return predictions_to_write


def write_dataframe_to_csv(folder, filename, data_to_write):
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
    data_to_write.to_csv(filepath, encoding='utf-8', index=False)

    return filepath


def process_input_arguments():
    parser = argparse.ArgumentParser('Import a model and classify images.')
    parser.add_argument('c1', help='Path of class 1 images')
    parser.add_argument('c2', help='Path of class 2 images')
    parser.add_argument('models', help='Folder of models to use')
    color_mode_group = parser.add_mutually_exclusive_group()
    color_mode_group.add_argument('-color', action='store_true', help='Images are in RGB color mode. (Default)')
    color_mode_group.add_argument('-bw', action='store_true', help='Images are in grayscale color mode.')
    args = parser.parse_args()

    image_folders = (args.c1, args.c2)
    class_labels: tuple = parse_class_names_from_image_folders(image_folders)
    list_of_models = None
    if ".model" in args.models:
        list_of_models = [args.models]
    else:
        list_of_models = os.listdir(args.models)
        list_of_models = [(args.models + os.path.sep + filename) for filename in list_of_models]

    return image_folders, class_labels, list_of_models


if __name__ == '__main__':
    main()
