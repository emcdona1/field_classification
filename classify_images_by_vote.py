import os
import argparse
import pandas as pd
import tensorflow as tf
from utilities.timer import Timer
import numpy as np
from datetime import datetime
from labeled_images.labeledimages import LabeledImages
from labeled_images.colormode import ColorMode
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

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

    # Seu up dataframe
    combined_results = pd.DataFrame()
    combined_results['filename'] = images.test_img_names
    all_predictions = pd.DataFrame()

    # Set up prediction list
    predictions = []
    col = 0
    row = 0

    # Return predictions
    for model_path in list_of_models:
        predictions, col, row = classify_images_with_a_model_multiclass(images.class_labels, all_predictions, images,
                                                                        model_path)

    # Find average of all predictions per image
    mean = []
    for x in range(row):
        total = 0
        for y in range(col):
            total = total + predictions[x][y]
        mean.append(total / row)

    # add to combined_results
    combined_results[r'saved_models\CNN_1.model'] = mean
    combined_results['voted_probability'] = mean
    combined_results['actual_class'] = images.test_labels

    # Store actual prediction per image
    actual_predicts = []
    cls = 0
    for i in range(row):
        max_val = 0
        for j in range(col):
            if predictions[i][j] > max_val:
                max_val = predictions[i][j]
                cls = j
        guess = cls
        actual_predicts.append(guess)
    combined_results['voted_label'] = actual_predicts

    # Add image labels
    labels = []
    for x in range(col):
        labels.append(x)

    # Generate confusion matrix
    matrix = confusion_matrix(images.test_labels, actual_predicts, labels=labels)
    display_matrix = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display_matrix.plot()

    # Label combined_results
    combined_results.columns = ['filename', r'saved_models\CNN_1.model', 'voted_probability', 'actual_class',
                                'voted_label']

    # Generate CVS file
    if not os.path.exists('predictions'):
        os.makedirs('predictions')
    write_dataframe_to_csv('predictions', 'model_vote_predict', combined_results)

    # Finish execution
    timer.stop()
    timer.print_results()

    # Show accuracy score and confusion matrix
    acc = accuracy_score(images.test_labels, actual_predicts)
    print(f'The final accuracy is: {acc}')
    plt.show()


def classify_images_with_a_model_multiclass(class_labels: list, combined_results: pd.DataFrame,
                                            images: LabeledImages, model_path: str):
    model_name = os.path.basename(model_path)
    if ".model" in model_path:

        # Load model
        model = tf.keras.models.load_model(model_path)
        print('Model ' + model_name + ' loaded.')

        # Generate predictions and label results
        predictions: np.array = model.predict(images.test_image_set)
        # test_dataset = tf.data.Dataset.from_tensor_slices([images.test_features])
        # predictions_using_from_tensor_slices_method = model.predict(test_dataset)

        # Store class labels
        clslst = []
        for x in range(predictions.shape[1]):
            clslst.append(images.class_labels[x] + '_prediction')

        # label predictions
        headers = clslst
        predictions = pd.DataFrame(predictions, columns=headers)
        print('Predictions generated.')

        # get prediction architecture
        count_row = predictions.shape[0]
        count_col = predictions.shape[1]

        # Create prediction list
        predict_list = []
        for x in range(count_row):
            for y in range(0, count_col):
                predict_list.append(predictions.at[x, images.class_labels[y] + '_prediction'])
        predict_group_list = [predict_list[i:i + count_col] for i in range(0, len(predict_list), count_col)]
        return predict_group_list, count_col, count_row
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
    parser.add_argument('img_size', type=int, help='Image dimension (one side in pixels -- square image assumed).')
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
    image_size = args.img_size

    return image_folders, list_of_models, color_mode, image_size


if __name__ == '__main__':
    main()
