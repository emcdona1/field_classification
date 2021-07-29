import argparse
import os
from labeled_images.colormode import ColorMode
from pathlib import Path
from typing import Tuple


class CNNArguments:
    def __init__(self):
        self._parser = argparse.ArgumentParser(
            'Create and train CNNs for binary classification of images, using cross-fold validation.')
        self._args: argparse.Namespace = self.set_up_parser_arguments()

        self.training_image_folder: Path = self._validate_training_folder()
        self.image_size: Tuple[int, int] = self._validate_image_size()
        self.color_mode: ColorMode = self._set_color_mode()
        self.lr: float = self._validate_learning_rate()
        self.n_folds: int = self._validate_n_folds()
        self.n_epochs: int = self._validate_n_epochs()
        self.batch_size: int = self._validate_batch_size()

    def set_up_parser_arguments(self):
        # folder argument
        self._parser.add_argument('training_set', help='Directory containing training images.')

        # image arguments
        self._parser.add_argument('height', type=int, default=256, help='Desired image height.')
        self._parser.add_argument('-w', '--width', type=int, help='Desired image width. (Omit for square images)')
        color_mode_group = self._parser.add_mutually_exclusive_group()
        color_mode_group.add_argument('-color', action='store_true', help='Images are in RGB color mode. (Default)')
        color_mode_group.add_argument('-bw', action='store_true', help='Images are in grayscale color mode.')

        # model training arguments
        self._parser.add_argument('-lr', '--learning_rate', type=float,
                                  default=0.001, help='Learning rate for training. (Default = 0.001)')
        self._parser.add_argument('-f', '--n_folds', type=int, default=1,
                                  help='Number of folds (minimum 1) for cross validation. (Default = 1)')
        self._parser.add_argument('-e', '--n_epochs', type=int, default=25,
                                  help='Number of epochs (minimum 5) per fold. (Default = 25)')
        self._parser.add_argument('-b', '--batch_size', type=int,
                                  default=64, help='Batch size (minimum 2) for training. (Default = 64)')
        return self._parser.parse_args()

def validate_required_arguments(args) -> (str, int):
    if not os.path.isdir(args.training_set):
        raise NotADirectoryError('Training set "%s" is not a valid directory path.' % args.training_set)
    if args.img_size <= 0:
        raise ValueError('Image size must be > 0. %i is not valid.' % args.img_size)
    return args.training_set, args.img_size


    def _set_color_mode(self) -> ColorMode:
        return ColorMode.grayscale if self._args.bw else ColorMode.rgb

    def _validate_learning_rate(self) -> float:
        lr = self._args.learning_rate
        if not 0 < lr <= 1:
            raise ValueError('Learning rate %f.6 is not valid. Must be in range 0 (exclusive) to 1 (inclusive).' % lr)
        return lr

    def _validate_n_folds(self) -> int:
        n_folds = self._args.n_folds
        if not n_folds >= 1:
            raise ValueError('%i is not a valid number of folds. Must be >= 1.' % n_folds)
        return n_folds

    def _validate_n_epochs(self) -> int:
        n_epochs = self._args.n_epochs
        if not n_epochs >= 5 or type(n_epochs) is not int:
            raise ValueError('# of epochs %i is not valid. Must be >= 5.)' % n_epochs)
        return n_epochs

    def _validate_batch_size(self) -> int:
        batch_size = self._args.batch_size
        if not batch_size >= 2:
            raise ValueError('Batch size %i is not valid. Must be >= 2.' % batch_size)
        return batch_size
