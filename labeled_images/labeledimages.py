import random
from labeled_images.colormode import ColorMode
import tensorflow as tf
from typing import List, Union, Tuple
import pandas as pd
import os
from utilities.dataloader import open_cv2_image, load_file_list_from_filesystem
from pathlib import Path
from models.cnn_lstm import CHAR_LIST

VALIDATION_SPLIT = 0.1  # todo: change this to 0.2?
MAX_LABEL_LENGTH = 50


def concurrently_shuffle_lists(list1, list2):
    conjoined_lists = list(zip(list1, list2))
    random.shuffle(conjoined_lists)
    list1, list2 = zip(*conjoined_lists)
    list1 = list(list1)
    list2 = list(list2)
    return list1, list2


# Map label string's characters to integer encodings
char_to_num = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=list(CHAR_LIST), mask_token=None
)

# Map label integer encodings back to original characters
num_to_char = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


class LabeledImages:
    def __init__(self, random_seed: int):
        self.seed: int = random_seed
        random.seed(self.seed)
        self.training_image_set: List[tf.data.Dataset] = [tf.data.Dataset.from_tensor_slices([0])]
        self.validation_image_set: List[tf.data.Dataset] = [tf.data.Dataset.from_tensor_slices([0])]
        self.img_count: int = 0
        self.batch_size: int = 0
        self.color_mode: ColorMode = ColorMode.rgb
        self.img_dimensions: tuple = (0, 0)
        self.class_labels: list = []
        self.n_folds: int = 1
        self.test_image_set: tf.data.Dataset = tf.data.Dataset.from_tensor_slices([0])
        self.test_img_names: list = list()
        self.test_features: list = list()
        self.test_labels: list = list()

    def load_training_images(self, training_images_location: Path,
                             image_size: Union[int, Tuple[int, int]],
                             color_mode: ColorMode = ColorMode.rgb,
                             shuffle=True,
                             n_folds=1,
                             batch_size=32,
                             metadata: Path = None) -> None:
        """ image_size = (height, width) or """
        if self.n_folds > 1:
            # TODO: implement splitting by folds for cross validation
            raise NotImplementedError('K-fold cross validation has not been implemented.')
        self.img_dimensions = image_size if type(image_size) is tuple else (image_size, image_size)
        self.color_mode = color_mode
        self.n_folds = n_folds
        self.batch_size = batch_size

        if not metadata:
            print('Fetching labels based on folder names.')
            self._load_images_based_on_directory_structure(training_images_location, shuffle)
        else:
            print('Fetching labels based on metadata CSV file.')
            self._load_images_based_on_metadata(training_images_location, shuffle, metadata)

        self.training_image_set[0] = self.training_image_set[0].cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        self.validation_image_set[0] = self.validation_image_set[0].cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    def _load_images_based_on_directory_structure(self, training_images_location, shuffle):
        self.training_image_set = [
            tf.keras.preprocessing.image_dataset_from_directory(training_images_location,
                                                                color_mode=self.color_mode.name,
                                                                image_size=self.img_dimensions,
                                                                seed=self.seed,
                                                                shuffle=shuffle,
                                                                batch_size=self.batch_size,
                                                                validation_split=VALIDATION_SPLIT,
                                                                subset='training')]
        self.validation_image_set = [
            tf.keras.preprocessing.image_dataset_from_directory(training_images_location,
                                                                color_mode=self.color_mode.name,
                                                                image_size=self.img_dimensions,
                                                                seed=self.seed,
                                                                shuffle=shuffle,
                                                                batch_size=self.batch_size,
                                                                validation_split=VALIDATION_SPLIT,
                                                                subset='validation')]
        self.class_labels = self.training_image_set[0].class_names
        images = os.walk(training_images_location)

        self.img_count = 0
        for d in images:
            files = [Path(f) for f in d[2]]
            image_files = [i for i in files if i.suffix in ['.jpeg', '.jpg', '.gif', '.png', '.bmp']]
            self.img_count += len(image_files)

    def _load_images_based_on_metadata(self, training_images_location, shuffle, metadata):
        training_images_location = load_file_list_from_filesystem(training_images_location)
        training_images_location = [i for i in training_images_location if '.csv' not in i]
        resize_image = tf.keras.layers.experimental.preprocessing.Resizing(self.img_dimensions[0],
                                                                           self.img_dimensions[1])
        image_list = list()
        for image_location in training_images_location:
            if self.color_mode == ColorMode.rgb:
                image = open_cv2_image(image_location)
            else:
                image = open_cv2_image(image_location, False)
                image = image.reshape((image.shape[0], image.shape[1], 1))
            image = resize_image(image)
            image_list.append(image)
        self.img_count = len(image_list)

        metadata = pd.read_csv(metadata)
        label_list = list()
        for image in training_images_location:
            query = os.path.basename(image)
            label = metadata[metadata['word_image_location'] == query]['human_transcription'].item()
            label_list.append(label)
        self.class_labels = list(set(label_list))

        if shuffle:
            image_list, label_list = concurrently_shuffle_lists(image_list, label_list)
        split = int(len(image_list) * VALIDATION_SPLIT)
        self.validation_image_set = [tf.data.Dataset.from_tensor_slices(
            (image_list[0:split], label_list[0:split]))]
        self.training_image_set = [tf.data.Dataset.from_tensor_slices(
            (image_list[split:], label_list[split:]))]

    def load_testing_images(self, testing_image_folder: str, image_size: int, color_mode: ColorMode = ColorMode.rgb):
        # todo: implement loading images with metadata
        self.color_mode = color_mode
        self.img_dimensions = (image_size, image_size)
        self.test_image_set = tf.keras.preprocessing.image_dataset_from_directory(testing_image_folder,
                                                                                  color_mode=self.color_mode.name,
                                                                                  image_size=self.img_dimensions,
                                                                                  shuffle=False)
        self.test_img_names = self.test_image_set.file_paths
        self.class_labels = self.test_image_set.class_names
        self.test_labels = list()
        self.test_features = list()
        for feature_class_pair in self.test_image_set.as_numpy_iterator():
            self.test_features = self.test_features + list(feature_class_pair[0])
            self.test_labels = self.test_labels + list(feature_class_pair[1])
