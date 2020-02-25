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
from model_training import ModelTrainer

matplotlib.use('Agg')  # required when running on server


def main() -> None:
    timer = Timer('Model training')
    class_labels, images, architecture, trainer, n_folds = setup()
    charts = Charts()

    # Train model
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for index, (training_idx_list, validation_idx_list) in enumerate(skf.split(images.features, images.labels)):
        # set up this model run
        architecture.reset_model()
        training_set = images.subset(training_idx_list)
        validation_set = images.subset(validation_idx_list)

        # train model
        history = trainer.train_blank_model(architecture, training_set, validation_set, index, n_folds)

        # validate newly created model
        validation_predicted_probability = architecture.model.predict_proba(validation_set[0])[:, 1]
        charts.update(history, index, validation_set[1], validation_predicted_probability, class_labels)

    finalize(charts, class_labels, timer)


def get_arguments():
    args = initialize_argparse()
    return validate_args(args)


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
    if not (image_directory == '') and not os.path.isdir(image_directory):
        raise NotADirectoryError(image_directory + ' is not a valid directory path.')

    class_labels = (args.c1, args.c2)
    if not os.path.isdir(class_labels[0]):
        raise NotADirectoryError(class_labels[0] + ' is not a valid directory path.')
    if not os.path.isdir(class_labels[1]):
        raise NotADirectoryError(class_labels[1] + ' is not a valid directory path.')

    img_size = args.img_size
    if not img_size > 0:
        raise ValueError('%i is not a valid image size (in pixels)' % img_size)

    color_mode = False if args.bw else True

    lr = args.learning_rate
    if not 0 < lr <= 1:
        raise ValueError('%f.6 is not a valid learning rate (0, 1]' % lr)

    n_folds = args.n_folds
    if not n_folds >= 2:
        raise ValueError('%i is not a valid number of folds (must be 2+)' % n_folds)

    n_epochs = args.n_epochs
    if not n_epochs >= 10:
        raise ValueError('%i is not a valid number of epochs (must be 10+)' % n_epochs)

    batch_size = args.batch_size
    if not batch_size >= 2:
        raise ValueError('%i is not a valid batch size (must be 2+)' % batch_size)

    return image_directory, class_labels, img_size, color_mode, lr, n_folds, n_epochs, batch_size


def setup():
    image_directory, class_labels, img_size, color_mode, lr, n_folds, n_epochs, batch_size = get_arguments()

    # create directories
    if not os.path.exists('graphs'):
        os.makedirs('graphs')
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    trainer = ModelTrainer(n_epochs, batch_size)

    # Load in images and shuffle order
    images = LabeledImages(image_directory, class_labels, color_mode, SEED)
    architecture = SmithsonianModel(img_size, color_mode, SEED, lr)

    return class_labels, images, architecture, trainer, n_folds


def split_image_sets(images, training_idx_list, validation_idx_list):
    train_features = images.features[training_idx_list]
    train_labels = images.labels[training_idx_list]
    validation_features = images.features[validation_idx_list]
    validation_labels = images.labels[validation_idx_list]
    return (train_features, train_labels), (validation_features, validation_labels)


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
