import argparse
import os
from labeled_images.colormode import ColorMode
from pathlib import Path
from typing import Tuple


class ModelTrainingArguments:
    def __init__(self):
        self._parser = argparse.ArgumentParser('Create and train NNs for image classification.')
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

    def _validate_training_folder(self) -> Path:
        training_path = Path(self._args.training_set)
        if not os.path.isdir(training_path):
            raise NotADirectoryError(f'"{self._args.training_set}" is not a valid directory path.')
        return training_path

    def _validate_image_size(self) -> (int, int):
        if self._args.width <= 0:
            raise ValueError(f'Image width must be > 0. {self._args.width} is not valid.')
        width = height = self._args.width
        if self._args.height:
            if self._args.height <= 0:
                raise ValueError(f'Image height must be > 0. {self._args.height} is not valid.')
            height = self._args.height
        return width, height

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
