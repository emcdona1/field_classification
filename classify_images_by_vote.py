import os
import argparse
from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from utilities.timer import Timer
from labeled_images.labeledimages import LabeledTestingImages
from labeled_images.colormode import ColorMode

THRESHOLD = 0.5
SEED = 1


def main():
    timer = Timer('Classify test images')
    image_folders, list_of_models, color_mode, image_size = process_input_arguments()

    images = LabeledTestingImages(SEED)
    images.load_testing_images(image_folders, image_size, color_mode)
    print('Images imported.')

    combined_results = pd.DataFrame()
    combined_results['filename'] = images.test_image_file_paths

    all_predictions = pd.DataFrame()
    # Set up prediction list
    predicted_probabilities = []
    num_classes = 0
    num_images = 0

    for model_path in list_of_models:
        predictions, num_classes, num_images = classify_images_with_a_model_multiclass(images, model_path)

    # # Find average of all predictions per image
    # mean = []
    # for x in range(num_images):
    #     total = 0
    #     for y in range(num_classes):
    #         total = total + predicted_probabilities[x][y]
    #     mean.append(total / num_images)
    #
    # # add to combined_results
    # combined_results['CNN_1.model'] = mean
    # combined_results['voted_probability'] = mean
    combined_results['actual_class'] = images.test_labels

    # Store actual prediction per image
    predicted_class = []
    cls = 0
    for i in range(num_images):
        max_val = 0
        for j in range(num_classes):
            if predicted_probabilities[i][j] > max_val:
                max_val = predicted_probabilities[i][j]
                cls = j
        predicted = cls
        predicted_class.append(predicted)
    combined_results['voted_label'] = predicted_class

    results_dir = 'test_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    generate_confusion_matrix(num_classes, images, predicted_class, results_dir)
    combined_results.columns = ['filename', 'CNN_1.model', 'voted_probability', 'actual_class', 'voted_label']
    combined_results.to_csv(str(Path(results_dir, 'combined_results.csv')))
    timer.stop()
    timer.print_results()

    prediction_accuracy = accuracy_score(images.test_labels, predicted_class)
    print(f'The final accuracy is: {prediction_accuracy}')

    exit(0)


def generate_confusion_matrix(col, images, predicted_class, results_dir: str):
    matrix = confusion_matrix(images.test_labels, predicted_class, labels=list(range(col)))
    display_matrix = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=list(range(col)))
    display_matrix.plot()

    file_name = 'test_confusion_matrix.png'
    plt.savefig(Path(results_dir, file_name), format='png')


def classify_images_with_a_model_multiclass(images: LabeledTestingImages, model_path: Union[str, Path]):
    model_name = Path(model_path).stem
    if Path(model_path).suffix == '.model':
        model = tf.keras.models.load_model(model_path)
        print(f'Model {model_name} loaded.')

        predictions: np.array = model.predict(images.test_image_set)

        labels = list(range(predictions.shape[1]))
        labels = [f'{images.class_labels[i]}_prediction' for i in labels]
        predictions = pd.DataFrame(predictions, columns=labels)
        print('Predictions generated.')

        count_row = predictions.shape[0]
        count_col = predictions.shape[1]

        predict_list = []
        for x in range(count_row):
            for y in range(0, count_col):
                predict_list.append(predictions.at[x, images.class_labels[y] + '_prediction'])
        predict_group_list = [predict_list[i:i + count_col] for i in range(0, len(predict_list), count_col)]
        return predict_group_list, count_col, count_row
    else:
        print(f'Model file path error: {model_path} is not a *.model file.')


def process_input_arguments():
    parser = argparse.ArgumentParser('Import a model and classify images.')
    parser.add_argument('images', help='Path containing folders of test images')
    parser.add_argument('img_size', type=int, help='Image dimension (one side in pixels -- square image assumed).')
    parser.add_argument('models', help='One model, or one folder of models to use.')
    color_mode_group = parser.add_mutually_exclusive_group()
    color_mode_group.add_argument('-color', action='store_true', help='Images are in RGB color mode. (Default)')
    color_mode_group.add_argument('-bw', action='store_true', help='Images are in grayscale color mode.')
    args = parser.parse_args()

    image_folders = args.images
    if '.model' in args.models:
        list_of_models = [args.models]
    else:
        list_of_models = os.listdir(args.models)
        list_of_models = [(args.models + os.path.sep + filename) for filename in list_of_models]
    color_mode = ColorMode.grayscale if args.bw else ColorMode.rgb
    image_size = args.img_size

    return image_folders, list_of_models, color_mode, image_size


if __name__ == '__main__':
    main()
