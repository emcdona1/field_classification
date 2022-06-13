from abc import abstractmethod, ABC
import argparse
import os
from labeled_images.colormode import ColorMode
from pathlib import Path
from typing import Tuple


class ModelArguments(ABC):
    def __init__(self):
        self._parser = argparse.ArgumentParser('Create and train NNs for image classification.')
        self._args: argparse.Namespace = self.set_up_parser_arguments()
        self.image_size: Tuple[int, int] = self._validate_image_size()
        self.color_mode: ColorMode = self._set_color_mode()

    @abstractmethod
    def set_up_parser_arguments(self):
        pass

    def _set_color_mode(self) -> ColorMode:
        return ColorMode.grayscale if self._args.bw else ColorMode.rgb

    def _validate_image_size(self) -> (int, int):
        assert self._args.height > 0, f'Image height must be > 0. {self._args.height} is not valid.'
        width = height = self._args.height
        if self._args.width:
            assert self._args.width > 0, f'Image width must be > 0. {self._args.width} is not valid.'
            width = self._args.width
        return height, width

    def _validate_image_folder(self, folder: str) -> Path:
        image_folder_path = Path(folder)
        assert os.path.isdir(image_folder_path), f'{folder} is not a valid directory path.'
        return image_folder_path


class ModelTrainingArguments(ModelArguments):
    def __init__(self):
        super().__init__()
        self.training_image_folder: Path = self._validate_image_folder(self._args.training_set)
        self.lr: float = self._validate_learning_rate()
        self.n_epochs: int = self._validate_n_epochs()
        self.batch_size: int = self._validate_batch_size()
        self.num_output_classes: int = self._validate_num_output_classes()

    def set_up_parser_arguments(self):
        self._parser.add_argument('training_set', help='Directory containing training images.')
        self._parser.add_argument('height', type=int, default=256, help='Desired image height.')
        self._parser.add_argument('-w', '--width', type=int, help='Desired image width. (Omit for square images)')
        color_mode_group = self._parser.add_mutually_exclusive_group()
        color_mode_group.add_argument('-color', action='store_true', help='Images are in RGB color mode. (Default)')
        color_mode_group.add_argument('-bw', action='store_true', help='Images are in grayscale color mode.')
        self._parser.add_argument('-lr', '--learning_rate', type=float,
                                  default=0.001, help='Learning rate for training. (Default = 0.001)')
        self._parser.add_argument('-e', '--n_epochs', type=int, default=25,
                                  help='Number of epochs (minimum 5) per fold. (Default = 25)')
        self._parser.add_argument('-b', '--batch_size', type=int,
                                  default=64, help='Batch size (minimum 2) for training. (Default = 64)')
        self._parser.add_argument('-cls', '--num_output_classes', default=2, type=int,
                                  help='Class size minimum 2, no max. (Default = 2)')
        return self._parser.parse_args()

    def _validate_learning_rate(self) -> float:
        lr = self._args.learning_rate
        assert 0 < lr < 1, f'Learning rate {lr:.6f} is not valid. Must be in range (0, 1).'
        return lr

    def _validate_n_epochs(self) -> int:
        n_epochs = self._args.n_epochs
        assert n_epochs >= 5, f'{n_epochs} is not a valid number of epochs. Must be >= 5.'
        return n_epochs

    def _validate_batch_size(self) -> int:
        batch_size = self._args.batch_size
        assert batch_size >= 2, f'{batch_size} is not a valid batch size. Must be >= 2.'
        return batch_size

    def _validate_num_output_classes(self) -> int:
        num_output_classes = self._args.num_output_classes
        assert num_output_classes >= 2, f'{num_output_classes} is not a valid number of classes. Must be >= 2.'
        return num_output_classes


class ModelTestingArguments(ModelArguments):
    def __init__(self):
        super().__init__()
        self.image_folders = self._validate_image_folder(self._args.images)
        self.list_of_models = self._validate_models()
        self.image_size = self._validate_image_size()

    def set_up_parser_arguments(self):
        self._parser.add_argument('images', help='Folder containing test images.')
        self._parser.add_argument('models', help='One model, or one folder of models to use.')
        self._parser.add_argument('height', type=int, default=256, help='Desired image height.')
        self._parser.add_argument('-w', '--width', type=int, help='Desired image width. (Omit for square images)')
        color_mode_group = self._parser.add_mutually_exclusive_group()
        color_mode_group.add_argument('-color', action='store_true', help='Images are in RGB color mode. (Default)')
        color_mode_group.add_argument('-bw', action='store_true', help='Images are in grayscale color mode.')
        return self._parser.parse_args()

    def _validate_models(self):
        if '.model' in self._args.models:
            list_of_models = [Path(self._args.models)]
        else:
            list_of_models = os.listdir(self._args.models)
            list_of_models = [Path(self._args.models, filename) for filename in list_of_models]
            list_of_models = [filename for filename in list_of_models if '.model' in str(filename)]
        assert len(list_of_models) >= 1, 'No .model files found.'
        for model_path in list_of_models:
            assert os.path.exists(model_path), f'{model_path} does not exist.'
        return list_of_models
