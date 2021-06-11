import os
import argparse

import numpy
import pandas as pd
import tensorflow as tf
from utilities.timer import Timer
import numpy as np
from datetime import datetime
from labeled_images.labeledimages import LabeledImages
from labeled_images.colormode import ColorMode

THRESHOLD = 0.5
SEED = 1


def main():
    # Start execution and parse arguments
    timer = Timer('Classifying a test set')
    image_folders, list_of_models, color_mode, image_size = process_input_arguments()

    # Import images
    images = LabeledImages(SEED)
    images.load_testing_images(image_folders, image_size, color_mode)
    print('Images imported.')

    combined_results = pd.DataFrame()
    combined_results['filename'] = images.test_img_names
    combined_results['actual_class'] = images.test_labels
    all_predictions = pd.DataFrame()

    for model_path in list_of_models:
        classify_images_with_a_model(images.class_labels, all_predictions, images, model_path)

    all_predictions['voted_probability'] = all_predictions.mean(axis=1)
    # calculate_confusion_matrix(combined_results)
    combined_results = combined_results.join(all_predictions)
    combined_results['tp'] = combined_results.eval('actual_class == 1 and voted_probability >= 0.5')
    combined_results['fn'] = combined_results.eval('actual_class == 1 and voted_probability < 0.5')
    combined_results['fp'] = combined_results.eval('actual_class == 0 and voted_probability >= 0.5')
    combined_results['tn'] = combined_results.eval('actual_class == 0 and voted_probability < 0.5')
    combined_results['voted_label'] = combined_results.eval('voted_probability >= 0.5')

    combined_results['tp'] = combined_results['tp'].map(lambda v: 1 if v else 0)
    combined_results['fn'] = combined_results['fn'].map(lambda v: 1 if v else 0)
    combined_results['fp'] = combined_results['fp'].map(lambda v: 1 if v else 0)
    combined_results['tn'] = combined_results['tn'].map(lambda v: 1 if v else 0)
    combined_results['voted_label'] = combined_results['voted_label'].map(lambda v: 1 if v else 0)

    combined_results.columns = ['filename', 'actual_class'] + list_of_models + \
                               ['voted_probability', 'tp', 'fn', 'fp', 'tn', 'voted_label']

    if not os.path.exists('predictions'):
        os.makedirs('predictions')
    write_dataframe_to_csv('predictions', 'model_vote_predict', combined_results)

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


def classify_images_with_a_model(class_labels: list, combined_results: pd.DataFrame,
                                 images: LabeledImages, model_path: str) -> None:
    model_name = os.path.basename(model_path)
    if ".model" in model_path:
        # Load model
        model = tf.keras.models.load_model(model_path)
        print('Model ' + model_name + ' loaded.')

        # Generate predictions and label results
        predictions: np.array = model.predict(images.test_image_set)
        test_dataset = tf.data.Dataset.from_tensor_slices([images.test_features])
        predictions_using_from_tensor_slices_method = model.predict(test_dataset)

        headers = [images.class_labels[0] + '_prediction', images.class_labels[1] + '_prediction']
        predictions = pd.DataFrame(predictions, columns=headers)
        print('Predictions generated.')

        # add newest predictions to results
        combined_results[model_name] = predictions[class_labels[1] + '_prediction']  # probability of class = 1
    else:
        print('Model file path error: "%s" is not a *.model file.' % model_path)


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
    parser.add_argument('images', help='Path containing folders of test images')
    parser.add_argument('size', type=int, help='Image dimension (one side in pixels -- square image assumed).')
    parser.add_argument('models', help='One model, or one folder of models to use.')
    color_mode_group = parser.add_mutually_exclusive_group()
    color_mode_group.add_argument('-color', action='store_true', help='Images are in RGB color mode. (Default)')
    color_mode_group.add_argument('-bw', action='store_true', help='Images are in grayscale color mode.')
    args = parser.parse_args()

    image_folders = args.images
    if ".model" in args.models:
        list_of_models = [args.models]
    else:
        list_of_models = os.listdir(args.models)
        list_of_models = [(args.models + os.path.sep + filename) for filename in list_of_models]
    color_mode = ColorMode.grayscale if args.bw else ColorMode.rgb
    image_size = args.size

    return image_folders, list_of_models, color_mode, image_size


if __name__ == '__main__':
    main()
