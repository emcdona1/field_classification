import argparse
import os
from labeled_images.colormode import ColorMode


class CNNArguments:
    def __init__(self):
        parser = argparse.ArgumentParser(
            'Create and train CNNs for binary classification of images, using cross-fold validation.')
        self.args: argparse.Namespace = self.set_up_parser_arguments(parser)

        self.image_folders = self.validate_image_folders()
        self.class_labels = self.parse_class_names_from_image_folders()
        self.color_mode = self.set_color_mode()
        self.lr = self.validate_learning_rate()
        self.n_folds = self.validate_n_folds()
        self.n_epochs = self.validate_n_epochs()
        self.batch_size = self.validate_batch_size()

    def set_up_parser_arguments(self, parser):
        # image arguments
        parser.add_argument('c1', help='Directory name containing images in class 1.')
        parser.add_argument('c2', help='Directory name containing images in class 2.')
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

    def validate_image_folders(self) -> (str, str):
        if not os.path.isdir(self.args.c1):
            raise NotADirectoryError('C1 value "%s" is not a valid directory path.' % self.args.c1)
        if not os.path.isdir(self.args.c2):
            raise NotADirectoryError('C2 value "%s" is not a valid directory path.' % self.args.c2)
        return self.args.c1, self.args.c2

    def parse_class_names_from_image_folders(self) -> (str, str):
        class1 = self.image_folders[0].strip(os.path.sep)
        class2 = self.image_folders[1].strip(os.path.sep)
        class1 = class1.split(os.path.sep)[class1.count(os.path.sep)]
        class2 = class2.split(os.path.sep)[class2.count(os.path.sep)]
        return class1, class2

    def set_color_mode(self):
        color_mode = ColorMode.BW if self.args.bw else ColorMode.RGB
        return color_mode

    def validate_learning_rate(self) -> float:
        lr = self.args.learning_rate
        if not 0 < lr <= 1:
            raise ValueError('Learning rate %f.6 is not valid. Must be in range 0 (exclusive) to 1 (inclusive).' % lr)
        return lr

    def validate_n_folds(self):
        n_folds = self.args.n_folds
        if not n_folds >= 1:
            raise ValueError('%i is not a valid number of folds. Must be >= 1.' % n_folds)
        return n_folds

    def validate_n_epochs(self) -> int:
        n_epochs = self.args.n_epochs
        if not n_epochs >= 10 or type(n_epochs) is not int:
            raise ValueError('# of epochs %i is not valid. Must be >= 10.)' % n_epochs)
        return n_epochs

    def validate_batch_size(self) -> int:
        batch_size = self.args.batch_size
        if not batch_size >= 2:
            raise ValueError('Batch size %i is not valid. Must be >= 2.' % batch_size)
        return batch_size
