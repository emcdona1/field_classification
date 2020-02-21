import os
import argparse
from timer import Timer
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import matplotlib
from image_handling import LabeledImages
from neural_network_models import SmithsonianModel
from data_and_visualization_io import Charts

matplotlib.use('Agg')  # required when running on server


def main() -> None:
    timer, args, charts = setup()

    # Load in images and shuffle order
    images = LabeledImages(args.dir, (args.c1, args.c2), args.color, SEED)

    # Train model
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)
    for index, (training_idx_list, validation_idx_list) in enumerate(skf.split(images.features, images.labels)):
        architecture = SmithsonianModel(args.img_size, color_mode=args.color, seed=SEED, lr=args.learning_rate)

        training_set, validation_set = split_image_sets(images, training_idx_list, validation_idx_list)
        history = train(index, architecture, training_set, validation_set, args.n_folds, args.batch_size, args.n_epochs)
        model = architecture.model

        validation_predicted_probability = model.predict_proba(validation_set[0])[:, 1]
        charts.update(history, index, validation_set[1], validation_predicted_probability, (args.c1, args.c2))

    finalize(charts, (args.c1, args.c2), timer)


def initialize_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        'Create and train CNNs for binary classification of images, using cross-fold validation.')
    # image arguments
    parser.add_argument('dir', default='', help='Base directory containing image to classify.')
    parser.add_argument('c1', help='Directory name containing images in class 1.')
    parser.add_argument('c2', help='Directory name containing images in class 2.')
    parser.add_argument('-s', '--img_size', type=int, default=256,
                        help='Image dimension in pixels (must be square).')
    color_mode_group = parser.add_mutually_exclusive_group()
    color_mode_group.add_argument('-color', action='store_true', help='(default) Images are in RGB color mode.')
    color_mode_group.add_argument('-bw', action='store_true', help='Images are in grayscale color mode.')

    # model creation argument
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='Learning rate for training.')

    # training run arguments
    parser.add_argument('-f', '--n_folds', type=int, default=10,
                        help='Number of folds (minimum 2) for cross validation.')
    parser.add_argument('-e', '--n_epochs', type=int, default=25, help='Number of epochs.')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size for training.')

    return parser.parse_args()


def validate_args(args: argparse.Namespace):
    image_directory = args.dir
    if not os.path.isdir(image_directory):
        raise NotADirectoryError(image_directory + ' is not a valid directory path.')
    class_labels = (args.c1, args.c2)
    if not os.path.isdir(class_labels[0]):
        raise NotADirectoryError(class_labels[0] + ' is not a valid directory path.')
    if not os.path.isdir(class_labels[1]):
        raise NotADirectoryError(class_labels[1] + ' is not a valid directory path.')

    # todo: continue to add data validation as needed
    # img_size, learning rate, folds, epochs, batch size

    return dir, class_labels


def setup():
    timer = Timer('Model training')
    args = initialize_argparse()
    dir, class_labels = validate_args(args)

    args.color = False if args.bw else True

    # create directories
    if not os.path.exists('graphs'):
        os.makedirs('graphs')
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    charts = Charts()
    return timer, args, charts


def split_image_sets(images, training_idx_list, validation_idx_list):
    train_features = images.features[training_idx_list]
    train_labels = images.labels[training_idx_list]
    validation_features = images.features[validation_idx_list]
    validation_labels = images.labels[validation_idx_list]
    return (train_features, train_labels), (validation_features, validation_labels)


def train(curr_fold, architecture, training_set, validation_set, n_folds, batch_size, n_epochs):
    print('Training model for fold %i of %i' % (curr_fold + 1, n_folds))
    # es_callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', \
    #        mode='min', min_delta = 0.05, patience = 20, restore_best_weights = True)
    history = architecture.model.fit(training_set[0], training_set[1], batch_size=batch_size, epochs=n_epochs,
                                     #        callbacks = [es_callback], \
                                     validation_data=(validation_set[0], validation_set[1]), verbose=2)
    architecture.model.save(os.path.join('saved_models', 'CNN_%i.model' % curr_fold + 1))

    return history


def finalize(charts, class_labels, timer):
    charts.finalize()
    # end
    print('class 1: ' + class_labels[0] + ', class 2: ' + class_labels[1])
    timer.stop()
    timer.results()


if __name__ == '__main__':
    # set up random seeds
    SEED = 1
    np.random.seed(SEED)
    tf.random.set_random_seed(SEED)
    random.seed(SEED)

    main()
