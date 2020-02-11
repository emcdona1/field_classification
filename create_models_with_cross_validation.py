import os
import argparse
from timer import Timer
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import matplotlib
from image_handling import import_images, LabeledImages
from neural_network_models import SmithsonianModel
from data_and_visualization_io import DataChartIO

matplotlib.use('Agg')  # required when running on server

# setup
SEED = 1
np.random.seed(SEED)
tf.compat.v1.random.set_random_seed(SEED)
random.seed(SEED)
BATCH_SIZE = 64
LEARNING_RATE = 0.0001


def main() -> None:
    # Set up
    timer = Timer('TrainingModels')
    parser = initialize_argparse()
    args = parser.parse_args()
    color = False if args.bw else True
    create_folders()

    # Load in images and shuffle order
    images = import_images(args.dir, (args.c1, args.c2), args.img_size, color, SEED)
    features = images.features
    labels = images.labels

    # Train model
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)
    charts = DataChartIO()
    for index, (training_idx_list, validation_idx_list) in enumerate(skf.split(features, labels)):
        model, history, validation_features, validation_labels = model_training(index, args.n_epochs, args.n_folds,
                                                                                images.features, images.labels,
                                                                                training_idx_list, validation_idx_list,
                                                                                args.img_size, color)
        model_validation(model, validation_features, charts, history, index, validation_labels)
    charts.save_results_to_csv()

    # end
    print('c1: ' + args.c1 + ', c2: ' + args.c2)
    timer.stop_timer()


def initialize_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        'Create and train CNNs for binary classification of images, using cross-fold validation.')
    parser.add_argument('dir', default='', help='Base directory containing image to classify.')
    parser.add_argument('c1', help='Directory name containing images in class 1')
    parser.add_argument('c2', help='Directory name containing images in class 2')
    parser.add_argument('-s', '--img_size', type=int, default=256,
                        help='Image dimension in pixels (must be square)')
    parser.add_argument('-f', '--n_folds', type=int, default=10,
                        help='Number of folds (minimum 2) for cross validation')
    parser.add_argument('-e', '--n_epochs', type=int, default=25, help='Number of epochs')

    color_mode_group = parser.add_mutually_exclusive_group()
    color_mode_group.add_argument('-color', action='store_true', help='(default) Images are in RGB color mode.')
    color_mode_group.add_argument('-bw', action='store_true', help='Images are in grayscale color mode.')

    return parser


def create_folders():
    if not os.path.exists('graphs'):
        os.makedirs('graphs')
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')


def model_training(curr_epoch, n_epochs, n_folds, features, labels, training_idx_list, validation_idx_list, img_size,
                   color):
    # set up training/validation
    train_features = features[training_idx_list]
    train_labels = labels[training_idx_list]
    validation_features = features[validation_idx_list]
    validation_labels = labels[validation_idx_list]
    architecture = SmithsonianModel(img_size, color_mode=color, seed=SEED, lr=LEARNING_RATE)

    print('Training model for fold' + str(curr_epoch + 1) + '/' + str(n_folds))
    # es_callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', \
    #        mode='min', min_delta = 0.05, patience = 20, restore_best_weights = True)
    history = architecture.model.fit(train_features, train_labels,
                                     batch_size=BATCH_SIZE, epochs=n_epochs,
                                     #        callbacks = [es_callback], \
                                     validation_data=(validation_features, validation_labels),
                                     verbose=2)
    architecture.model.save(os.path.join('saved_models', 'CNN_' + str(curr_epoch + 1) + '.model'))

    return architecture.model, history, validation_features, validation_labels


def model_validation(model, validation_features, charts, history, index, validation_labels):
    # Classify the validation set
    validation_predicted_probability = model.predict_proba(validation_features)[:, 1]
    validation_predicted_classification = [round(a + 0.0001) for a in validation_predicted_probability]
    charts.update_and_save_graphs(history, index, validation_labels,
                                  validation_predicted_classification, validation_predicted_probability)


if __name__ == '__main__':
    main()
