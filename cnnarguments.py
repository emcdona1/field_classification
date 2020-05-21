import argparse
import os


class CNNArguments:
    def __init__(self, parser: argparse.ArgumentParser):
        self.parser: argparse.ArgumentParser = parser

        # image arguments
        self.parser.add_argument('c1', help='Directory name containing images in class 1.')
        self.parser.add_argument('c2', help='Directory name containing images in class 2.')
        color_mode_group = self.parser.add_mutually_exclusive_group()
        color_mode_group.add_argument('-color', action='store_true', help='Images are in RGB color mode. (Default)')
        color_mode_group.add_argument('-bw', action='store_true', help='Images are in grayscale color mode.')

        # model creation argument
        self.parser.add_argument('-lr', '--learning_rate', type=float,
                                 default=0.0001, help='Learning rate for training. (Default = 0.0001)')

        # training run arguments
        self.parser.add_argument('-f', '--n_folds', type=int, default=10,
                                 help='Number of folds (minimum 1) for cross validation. (Default = 10)')
        self.parser.add_argument('-e', '--n_epochs', type=int, default=25,
                                 help='Number of epochs (minimum 10) per fold. (Default = 25)')
        self.parser.add_argument('-b', '--batch_size', type=int,
                                 default=64, help='Batch size (minimum 2) for training. (Default = 64)')

        self.args: argparse.Namespace = self.parser.parse_args()

    def image_folders_and_class_labels(self) -> tuple:
        if not os.path.isdir(self.args.c1):
            raise NotADirectoryError('%s is not a valid directory path.' % self.args.c1)
        if not os.path.isdir(self.args.c2):
            raise NotADirectoryError('%s is not a valid directory path.' % self.args.c2)
        image_folders = (self.args.c1, self.args.c2)
        class_labels = parse_class_names_from_image_folders(self.args)

        return image_folders, class_labels

    def learning_rate(self) -> float:
        lr = self.args.learning_rate
        if not 0 < lr <= 1:
            raise ValueError('%f.6 is not a valid learning rate. Must be in range 0 (exclusive) to 1 (inclusive).' % lr)
        return lr

    def color_mode(self) -> bool:
        return False if self.args.bw else True

    def n_folds(self) -> int:
        n_folds = self.args.n_folds
        if not n_folds >= 1:
            raise ValueError('%i is not a valid number of folds. Must be >= 1.' % n_folds)

        return n_folds

    def n_epochs(self) -> int:
        n_epochs = self.args.n_epochs
        if not n_epochs >= 10:
            raise ValueError('%i is not a valid number of epochs. Must be >= 10.)' % n_epochs)
        return n_epochs

    def batch_size(self) -> int:
        batch_size = self.args.batch_size
        if not batch_size >= 2:
            raise ValueError('%i is not a valid batch size. Must be >= 2.' % batch_size)
        return batch_size


def parse_class_names_from_image_folders(args) -> tuple:
    class1 = args.c1.strip(os.path.sep)
    class2 = args.c2.strip(os.path.sep)
    class1 = class1.split(os.path.sep)[class1.count(os.path.sep)]
    class2 = class2.split(os.path.sep)[class2.count(os.path.sep)]

    return class1, class2
