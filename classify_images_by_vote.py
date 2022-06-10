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
from models.modeltrainingarguments import ModelTestingArguments

THRESHOLD = 0.5
SEED = 1


def main():
    timer = Timer('Classify test images')
    arguments = ModelTestingArguments()

    images = LabeledTestingImages(SEED)
    images.load_testing_images(arguments.image_folders, arguments.image_size, arguments.color_mode)
    print('Images imported.')

    results_dir = 'test_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if len(arguments.list_of_models) == 1:
        predictions = predict_test_images(images, arguments.list_of_models[0])
        predicted_class = np.argmax(np.array(predictions), axis=1)

        combined_results = pd.DataFrame()
        combined_results['image filenames'] = images.test_image_file_paths
        combined_results['actual_class'] = images.test_labels
        combined_results['voted_label'] = predicted_class
        combined_results[predictions.columns] = predictions
        generate_confusion_matrix(len(images.class_labels), images, predicted_class, results_dir)
    else:
        predictions = list()
        predicted_class = list()
        for model_path in arguments.list_of_models:
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


if __name__ == '__main__':
    main()
