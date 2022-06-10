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

    results_dir = 'test_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if len(list_of_models) == 1:
        predictions = predict_test_images(images, list_of_models[0])
        predicted_class = np.argmax(np.array(predictions), axis=1)

        combined_results = pd.DataFrame()
        combined_results['image filenames'] = images.test_image_file_paths
        combined_results['actual_class'] = images.test_labels
        combined_results['voted_label'] = predicted_class
        combined_results[(predictions.columns)] = predictions
        generate_confusion_matrix(len(images.class_labels), images, predicted_class, results_dir)
    else:
        predictions = list()
        predicted_class = list()
        for model_path in list_of_models:
            next_predictions, num_classes = predict_test_images(images, model_path)
            predictions.append(next_predictions)
            predicted_class.append(np.argmax(np.array(predictions), axis=1))
            # todo: vote
        combined_results = pd.DataFrame()

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


def predict_test_images(images: LabeledTestingImages, model_path: Union[str, Path]) -> pd.DataFrame:
    model_name = Path(model_path).stem
    if Path(model_path).suffix == '.model':
        model = tf.keras.models.load_model(model_path)
        print(f'Model {model_name} loaded.')

        predictions: np.array = model.predict(images.test_image_set)

        labels = [f'{i}_prediction' for i in images.class_labels]
        print('Predictions generated.')

        predictions = pd.DataFrame(predictions, columns=labels)
        return predictions
    else:
        print(f'Model file path error: {model_path} is not a *.model file.')  # todo move this to argument processing


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
        list_of_models = [Path(args.models)]
    else:
        list_of_models = os.listdir(args.models)
        list_of_models = [Path(args.models, filename) for filename in list_of_models]
    color_mode = ColorMode.grayscale if args.bw else ColorMode.rgb
    image_size = (args.img_size, args.img_size)

    return image_folders, list_of_models, color_mode, image_size


if __name__ == '__main__':
    main()
