import os
import argparse
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import matplotlib
from image_importer import ImageLoader
from smithsonian_model import SmithsonianModel
from data_plotter import ChartCreator

matplotlib.use('Agg')  # required when running on server

# setup
SEED = 1
np.random.seed(SEED)
tf.compat.v1.random.set_random_seed(SEED)
random.seed(SEED)
BATCH_SIZE = 64
LEARNING_RATE = 0.0001


def parse_arguments():
    parser = argparse.ArgumentParser('import images and train model')
    parser.add_argument('-d', '--directory', default='', help='Folder holding category folders')
    parser.add_argument('-c1', '--category1', help='Folder of class 1')
    parser.add_argument('-c2', '--category2', help='Folder of class 2')
    parser.add_argument('-s', '--img_size', default=256, help='Image dimension in pixels')
    parser.add_argument('-cm', '--color_mode', default=1, help='Color mode to use (1=color, 0=grayscale)')
    parser.add_argument('-n', '--number_folds', default=10, help='Number of folds (minimum 2) for cross validation')
    parser.add_argument('-e', '--number_epochs', default=25, help='Number of epochs')
    args = parser.parse_args()

    img_directory = args.directory
    folders = [args.category1, args.category2]
    img_size = int(args.img_size)
    n_folds = int(args.number_folds)
    n_epochs = int(args.number_epochs)
    color = True if int(args.color_mode) == 1 else False

    return n_folds, img_directory, folders, img_size, color, n_epochs


def create_folders():
    if not os.path.exists('graphs'):
        os.makedirs('graphs')
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

def train_model_on_images(model, train_features, train_labels, num_epochs, val_features, val_labels):
    print("Training model")
    # es_callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', \
    #        mode='min', min_delta = 0.05, patience = 20, restore_best_weights = True)
    history = model.fit(train_features, train_labels,
                        batch_size=BATCH_SIZE, epochs=num_epochs,
                        #        callbacks = [es_callback], \
                        validation_data=(val_features, val_labels),
                        verbose=2)
    return history


def save_results_to_csv(results):
    # results = results.rename({0: 'Fold Number',
    #                           1: 'Training Loss',
    #                           2: 'Training Accuracy',
    #                           3: 'Validation Loss',
    #                           4: 'Validation Accuracy',
    #                           5: 'True Negatives', 6: 'False Positives',
    #                           7: 'False Negatives', 8: 'True Positives'})
    results.to_csv(os.path.join('graphs', 'final_acc_loss.csv'), encoding='utf-8', index=False)


def train_cross_validate(n_folds, features, labels, img_size, color, num_epochs):
    """ Import images from the file system and returns two numpy arrays containing the pixel information and classification.

    Parameters:
    -----
    @ n_folds : int
    Number of folds to train on (minimum of 2)

    @ features : np.array
    Array containing the pixel values -- dimensions are (img_size x img_size x 3) if color,
    (img_size x img_size x 1) if grayscale.

    @ labels : np.array
    Array containing the actual labels (0/1) of the images

    @ img_size : int
    Pixel dimensions of images

    @ num_epochs : int
    Maximum number of epochs to perform on each k-fold.

    Output:
    -----
    none
    """
    charts = ChartCreator()

    # initialize stratifying k fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    # data frame to save values of loss and validation after each fold
    results = pd.DataFrame(columns=['Fold Number', 'Training Loss', 'Training Accuracy',
                                    'Validation Loss', 'Validation Accuracy',
                                    'True Negatives', 'False Positives', 'False Negatives', 'True Positives'])

    for index, (train_indices, val_indices) in enumerate(skf.split(features, labels)):
        print("Training on fold " + str(index + 1) + "/" + str(n_folds))
        train_features = features[train_indices]
        train_labels = labels[train_indices]
        # print("Training data obtained")
        validation_features = features[val_indices]
        validation_labels = labels[val_indices]
        # print("Validation data obtained")

        # Create new model each time
        architecture = SmithsonianModel(img_size, color_mode=color, seed=SEED, lr=LEARNING_RATE)
        history = train_model_on_images(architecture.model, train_features, train_labels,
                                        num_epochs, validation_features, validation_labels)

        architecture.model.save(os.path.join('saved_models', 'CNN_' + str(index + 1) + '.model'))

        charts.plot_accuracy(history, index)
        charts.plot_loss(history, index)

        # Classify the validation set
        validation_predicted_probability = architecture.model.predict_proba(validation_features)[:, 1]
        validation_predicted_classification = [round(a + 0.0001) for a in validation_predicted_probability]

        # Plot ROC curve, generate confusion matrix
        charts.plot_roc_curve(index, validation_labels, validation_predicted_probability)
        tn, fp, fn, tp = charts.generate_confusion_matrix(validation_labels, validation_predicted_classification)

        num_epochs = len(history.history['loss'])
        # save the stats of the last epoch (i.e. end of the fold) to the results file
        results = results.append([[index + 1,
                                   history.history['loss'][num_epochs - 1],
                                   history.history['acc'][num_epochs - 1],
                                   history.history['val_loss'][num_epochs - 1],
                                   history.history['val_acc'][num_epochs - 1],
                                   tn, fp, fn, tp]])

    save_results_to_csv(results)


if __name__ == '__main__':
    start_time = time.time()

    # Set up
    n_folds, img_directory, folders, img_size, color, n_epochs = parse_arguments()
    create_folders()

    # Load in images and shuffle order
    images = ImageLoader(img_directory, folders, img_size, color, SEED)
    features = images.features
    labels = images.labels

    # Train model
    train_cross_validate(n_folds, features, labels, img_size, color, n_epochs)

    # end
    print('c1: ' + folders[0] + ', c2: ' + folders[1])
    end_time = time.time()
    print('Completed in %.1f seconds' % (end_time - start_time))
