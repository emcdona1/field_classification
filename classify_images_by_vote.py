import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from utilities.timer import Timer
from utilities.dataloader import save_dataframe_as_csv
from labeled_images.labeledimages import LabeledTestingImages
from models.modeltrainingarguments import ModelTestingArguments


THRESHOLD = 0.5


def main():
    timer = Timer('Classify test images')
    arguments = ModelTestingArguments()

    images = LabeledTestingImages()
    images.load_testing_images(arguments.image_folders, arguments.image_size, arguments.color_mode)
    print('Images imported.')

    results_dir = Path('test_results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    combined_results = pd.DataFrame()
    combined_results['image filenames'] = images.test_image_file_paths
    sum_of_predictions = np.zeros((images.img_count, len(images.class_labels)))
    for model_path in arguments.list_of_models:
        predicted_class, current_predictions = make_and_save_predictions(model_path, combined_results, images)
        generate_confusion_matrix(len(images.class_labels), images, predicted_class, model_path.stem, results_dir)
        sum_of_predictions = sum_of_predictions + current_predictions
    if len(arguments.list_of_models) > 1:
        votes = np.argmax(np.array(sum_of_predictions), axis=1)
        generate_confusion_matrix(len(images.class_labels), images, votes, 'vote', results_dir)
        combined_results['voted'] = votes
        voted_labels = pd.DataFrame(votes)
        voted_labels = voted_labels.apply(lambda a: images.class_labels[int(a)], axis=1)
        combined_results['voted_class'] = voted_labels

    save_dataframe_as_csv(results_dir, 'results', combined_results, timestamp=False)
    timer.stop()
    timer.print_results()

    prediction_accuracy = accuracy_score(images.test_labels, predicted_class)
    print(f'The final accuracy is: {prediction_accuracy}')

    exit(0)

def make_and_save_predictions(model_path: Path,
                              combined_results: pd.DataFrame,
                              images: LabeledTestingImages) -> (np.ndarray, np.ndarray):
    model_name = model_path.stem
    model = tf.keras.models.load_model(model_path)
    print(f'Model {model_path.stem} loaded.')
    predictions: np.array = model.predict(images.test_image_set)
    predicted_class = np.argmax(np.array(predictions), axis=1)
    print('Predictions generated.')

    labels = [f'{i} prediction' for i in images.class_labels]
    labeled_predictions = pd.DataFrame(predictions, columns=labels)

    prediction_column_names = [f'{model_name} {i}' for i in list(labeled_predictions.columns)]
    combined_results[prediction_column_names] = labeled_predictions
    combined_results[f'{model_name} - actual class'] = images.test_labels
    combined_results[f'{model_name} - voted label'] = predicted_class

    return predicted_class, predictions


def generate_confusion_matrix(col, images, predicted_class, matrix_name: str, results_dir: Path) -> None:
    matrix = confusion_matrix(images.test_labels, predicted_class, labels=list(range(col)))
    display_matrix = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=list(range(col)))
    display_matrix.plot()
    file_name = f'test_set-confusion_matrix-{matrix_name}.png'
    plt.savefig(Path(results_dir, file_name), format='png')


if __name__ == '__main__':
    main()
