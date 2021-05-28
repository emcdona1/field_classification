import argparse
import os
from labeled_images.colormode import ColorMode
from typing import Tuple


class CNNArguments:
    def __init__(self):
        # todo: TEMP
        self.training_image_folder = 'mini_test_set'
        self.testing_image_folder = 'mini_test_set'
        self.image_size = 256
        self.n_folds = 1
        self.batch_size = 32
        self.n_epochs = 5
        self.color_mode = ColorMode.RGB
        self.lr = 0.001
        # todo: /TEMP
        """parser = argparse.ArgumentParser(
            'Create and train CNNs for binary classification of images, using cross-fold validation.')
        args: argparse.Namespace = set_up_parser_arguments(parser)

        # todo: remove image_folders
        self.image_folders, self.training_image_folder, self.testing_image_folder, self.image_size = \
            validate_required_arguments(args)

        # todo: remove class_labels and get from NewLabeledImages
        self.class_labels = parse_class_names_from_image_folders(self.image_folders)
        self.color_mode = set_color_mode(args)
        self.lr = validate_learning_rate(args)
        self.n_folds = validate_n_folds(args)
        self.n_epochs = validate_n_epochs(args)
        self.batch_size = validate_batch_size(args)"""


def set_up_parser_arguments(parser):
    # new arguments:
    parser.add_argument('training_set', help='Directory containing training/validation images.')
    parser.add_argument('testing_set', help='Directory containing testing images.')  # todo: make this optional
    parser.add_argument('img_size', type=int, help='Desired image width/height (square images).')
    # rest of arguments are unchanged

    # image arguments
    parser.add_argument('c1', help='Directory name containing images in class 1.')  # todo: remove
    parser.add_argument('c2', help='Directory name containing images in class 2.')  # todo: remove
    color_mode_group = parser.add_mutually_exclusive_group()
    color_mode_group.add_argument('-color', action='store_true', help='Images are in RGB color mode. (Default)')
    color_mode_group.add_argument('-bw', action='store_true', help='Images are in grayscale color mode.')
    # model creation argument
    parser.add_argument('-lr', '--learning_rate', type=float,
                        default=0.0001, help='Learning rate for training. (Default = 0.0001)')
    # training run arguments
    parser.add_argument('-f', '--n_folds', type=int, default=10,
                        help='Number of folds (minimum 1) for cross validation. (Default = 10)')
    parser.add_argument('-e', '--n_epochs', type=int, default=25,
                        help='Number of epochs (minimum 10) per fold. (Default = 25)')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=64, help='Batch size (minimum 2) for training. (Default = 64)')
    return parser.parse_args()


def parse_class_names_from_image_folders(image_folders) -> (str, str):
    class1 = image_folders[0].strip(os.path.sep)
    class2 = image_folders[1].strip(os.path.sep)
    class1 = class1.split(os.path.sep)[class1.count(os.path.sep)]
    class2 = class2.split(os.path.sep)[class2.count(os.path.sep)]
    return class1, class2


def validate_required_arguments(args) -> (Tuple[str, str], str, str, int):
    if not os.path.isdir(args.training_set):
        raise NotADirectoryError('Training set "%s" is not a valid directory path.' % args.training_set)
    if not os.path.isdir(args.training_set):
        raise NotADirectoryError('Training set "%s" is not a valid directory path.' % args.training_set)
    if args.img_size <= 0:
        raise ValueError('Image size must be > 0. %i is not valid.' % args.img_size)
    # todo: remove c1, c2
    if not os.path.isdir(args.c1):
        raise NotADirectoryError('C1 value "%s" is not a valid directory path.' % args.c1)
    if not os.path.isdir(args.c2):
        raise NotADirectoryError('C2 value "%s" is not a valid directory path.' % args.c2)
    return (args.c1, args.c2), args.training_set, args.testing_set, args.img_size


def set_color_mode(args):
    color_mode = ColorMode.BW if args.bw else ColorMode.RGB
    return color_mode


def validate_learning_rate( args) -> float:
    lr = args.learning_rate
    if not 0 < lr <= 1:
        raise ValueError('Learning rate %f.6 is not valid. Must be in range 0 (exclusive) to 1 (inclusive).' % lr)
    return lr


def validate_n_folds( args):
    n_folds = args.n_folds
    if not n_folds >= 1:
        raise ValueError('%i is not a valid number of folds. Must be >= 1.' % n_folds)
    return n_folds


def validate_n_epochs( args) -> int:
    n_epochs = args.n_epochs
    if not n_epochs >= 10 or type(n_epochs) is not int:
        raise ValueError('# of epochs %i is not valid. Must be >= 10.)' % n_epochs)
    return n_epochs


def validate_batch_size( args) -> int:
    batch_size = args.batch_size
    if not batch_size >= 2:
        raise ValueError('Batch size %i is not valid. Must be >= 2.' % batch_size)
    return batch_size
