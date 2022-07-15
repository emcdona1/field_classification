import os
from pathlib import Path
import tensorflow as tf
import random
from typing import List, Union, Tuple
from labeled_images.colormode import ColorMode

VALIDATION_SPLIT = 0.1


class LabeledImages:
    def __init__(self, random_seed: int):
        self.seed: int = random_seed
        random.seed(self.seed)
        self.training_image_set: tf.data.Dataset = tf.data.Dataset.from_tensor_slices([0])
        self.validation_image_set: tf.data.Dataset = tf.data.Dataset.from_tensor_slices([0])
        self.img_count: int = 0
        self.batch_size: int = 0
        self.color_mode: ColorMode = ColorMode.rgb
        self.img_dimensions: tuple = (0, 0)
        self.class_labels: list = []

    def load_training_images(self, training_images_location: Union[str, Path],
                             image_size: Union[int, Tuple[int, int]],
                             color_mode: ColorMode = ColorMode.rgb,
                             shuffle=True,
                             batch_size=32) -> None:
        """ image_size = (height, width) or single int for square images """
        self.img_dimensions = image_size if type(image_size) is tuple else (image_size, image_size)
        self.color_mode = color_mode
        self.batch_size = batch_size

        print('Fetching labels based on folder names.')
        self._load_images_based_on_directory_structure(training_images_location, shuffle)

        self.training_image_set = self.training_image_set.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        self.validation_image_set = self.validation_image_set.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    def _load_images_based_on_directory_structure(self, training_images_location, shuffle):
        self.training_image_set = tf.keras.preprocessing.image_dataset_from_directory(training_images_location,
                                                                                      color_mode=self.color_mode.name,
                                                                                      image_size=self.img_dimensions,
                                                                                      seed=self.seed,
                                                                                      shuffle=shuffle,
                                                                                      batch_size=self.batch_size,
                                                                                      validation_split=VALIDATION_SPLIT,
                                                                                      subset='training')
        self.validation_image_set = tf.keras.preprocessing.image_dataset_from_directory(training_images_location,
                                                                                        color_mode=self.color_mode.name,
                                                                                        image_size=self.img_dimensions,
                                                                                        seed=self.seed,
                                                                                        shuffle=shuffle,
                                                                                        batch_size=self.batch_size,
                                                                                        validation_split=VALIDATION_SPLIT,
                                                                                        subset='validation')
        self.class_labels = self.training_image_set.class_names

        self.img_count = 0
        self.count_per_class = {}
        count = 0
        for parent, child, files in os.walk(training_images_location):
            image_files = [Path(i) for i in files if Path(i).suffix in ['.jpeg', '.jpg', '.gif', '.png', '.bmp']]
            if len(image_files) > 0:
                self.count_per_class[count] = len(image_files)
                count += 1
                self.img_count += len(image_files)


class LabeledTestingImages(LabeledImages):
    def __init__(self):
        super().__init__(1)
        self.test_image_set: tf.data.Dataset = tf.data.Dataset.from_tensor_slices([0])
        self.test_image_file_paths: list = []
        self.test_features: list = []
        self.test_labels: list = []

    def load_testing_images(self, testing_images_location: Union[str, Path],
                            image_size: Tuple[int, int],
                            color_mode: ColorMode = ColorMode.rgb):
        self.img_dimensions = image_size
        self.color_mode = color_mode
        self.test_image_set = tf.keras.preprocessing.image_dataset_from_directory(testing_images_location,
                                                                                  color_mode=self.color_mode.name,
                                                                                  image_size=self.img_dimensions,
                                                                                  shuffle=False)
        self.test_image_file_paths = self.test_image_set.file_paths
        self.class_labels = self.test_image_set.class_names
        self.test_labels = list()
        self.test_features = list()
        for feature_class_pair in self.test_image_set.as_numpy_iterator():
            self.test_features = self.test_features + list(feature_class_pair[0])
            self.test_labels = self.test_labels + list(feature_class_pair[1])

        self.img_count = 0
        for parent, child, files in os.walk(testing_images_location):
            image_files = [Path(i) for i in files if Path(i).suffix in ['.jpeg', '.jpg', '.gif', '.png', '.bmp']]
            self.img_count += len(image_files)
