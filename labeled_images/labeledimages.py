import os
from pathlib import Path
import tensorflow as tf
import pandas as pd
import numpy as np
import random
from typing import List, Union, Tuple
from utilities.dataloader import load_file_list_from_filesystem
from labeled_images.colormode import ColorMode

VALIDATION_SPLIT = 0.1  # todo: change this to 0.2?
MAX_LABEL_LENGTH = 50
CHAR_LIST: str = '\' !"#&()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'


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

    def load_training_images(self, training_images_location: Union[str, Path],
                             image_size: Union[int, Tuple[int, int]],
                             color_mode: ColorMode = ColorMode.rgb,
                             shuffle=True,
                             n_folds=1,
                             batch_size=32,
                             metadata: Union[str, Path] = None) -> None:
        """ image_size = (height, width) or """
        if self.n_folds > 1:
            # TODO: implement splitting by folds for cross validation
            raise NotImplementedError('K-fold cross validation has not been implemented.')
        self.img_dimensions = image_size if type(image_size) is tuple else (image_size, image_size)
        self.color_mode = color_mode
        self.n_folds = n_folds
        self.batch_size = batch_size

        if metadata:  # todo: change from file path to boolean (then search for 'csv glob in image folder)
            print('Fetching labels based on metadata CSV file.')
            self._load_images_based_on_metadata(training_images_location, shuffle, metadata)
        else:
            print('Fetching labels based on folder names.')
            self._load_images_based_on_directory_structure(training_images_location, shuffle)

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
        training_images_location: List[Path] = load_file_list_from_filesystem(training_images_location)
        training_images_location: List[str] = [str(i) for i in training_images_location if not i.suffix == '.csv']
        image_list = np.array(training_images_location)
        self.img_count = len(image_list)
        image_list = np.array(image_list)

        metadata = pd.read_csv(metadata)
        label_list = list()
        for image in training_images_location:
            query = os.path.basename(image)
            label = metadata[metadata['word_image_location'] == query]['human_transcription'].item()
            label_list.append(label)
        self.class_labels = list(set(label_list))
        label_list = [e.ljust(MAX_LABEL_LENGTH) for e in label_list]
        label_list = np.array(label_list)

        def split_data(images, labels, train_split, shuffle_images):
            # 1. Get the total size of the dataset
            size = len(images)
            # 2. Make an indices array and shuffle it, if required
            indices = np.arange(size)
            if shuffle_images:
                np.random.shuffle(indices)
            # 3. Get the size of training samples
            split = int(size * train_split)
            # 4. Split data into training and validation sets
            train_features, validation_features  = images[indices[0:split]], labels[indices[0:split]]
            train_labels, validation_labels = images[indices[split:]], labels[indices[split:]]
            return train_features, validation_features, train_labels, validation_labels

        train_features, train_labels, val_features, val_labels = split_data(image_list, label_list, 1-VALIDATION_SPLIT, shuffle)

        def encode_one_image_and_labels(img_path, img_label) -> (tf.Tensor, str):
            img = tf.io.read_file(img_path)
            img = tf.io.decode_jpeg(img, channels=1)
            # if Path(img_path).suffix == '.gif':
            #     # todo: handle GIFs (4D)
            #     pass
            img = tf.image.convert_image_dtype(img, tf.float32)  # converts to [0, 1]
            img = tf.image.resize(img, list(self.img_dimensions))
            img = tf.transpose(img, perm=[1, 0, 2])
            img_label = char_to_num(tf.strings.unicode_split(img_label, input_encoding='UTF-8'))
            return {'image': img, 'label': img_label}
        self.training_image_set = [tf.data.Dataset.from_tensor_slices((train_features, train_labels))]
        self.training_image_set[0] = self.training_image_set[0] \
            .map(encode_one_image_and_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .batch(self.batch_size )\
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        self.validation_image_set = [tf.data.Dataset.from_tensor_slices((val_features, val_labels))]
        self.validation_image_set[0] = self.validation_image_set[0] \
            .map(encode_one_image_and_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .batch(self.batch_size) \
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

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
