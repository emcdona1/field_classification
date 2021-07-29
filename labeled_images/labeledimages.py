import random
from labeled_images.colormode import ColorMode
import tensorflow as tf
from typing import List, Union, Tuple


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

    def load_images_from_folders(self, training_images_location: str, image_size: Union[int, Tuple[int, int]],
                                 color_mode: ColorMode = ColorMode.rgb, shuffle=True, n_folds=1, batch_size=32) -> None:
        self.color_mode = color_mode
        if image_size is int:
            self.img_dimensions = (image_size, image_size)
        else:
            self.img_dimensions = image_size
        self.n_folds = n_folds
        self.batch_size = batch_size
        if self.n_folds > 1:
            # TODO: implement splitting by folds for cross validation
            pass
        else:
            self.training_image_set = list()
            self.training_image_set.append(
                tf.keras.preprocessing.image_dataset_from_directory(training_images_location,
                                                                    color_mode=self.color_mode.name,
                                                                    image_size=self.img_dimensions,
                                                                    seed=self.seed,
                                                                    shuffle=shuffle,
                                                                    batch_size=self.batch_size,
                                                                    validation_split=0.1,
                                                                    subset='training'))
            self.validation_image_set = list()
            self.validation_image_set.append(
                tf.keras.preprocessing.image_dataset_from_directory(training_images_location,
                                                                    color_mode=self.color_mode.name,
                                                                    image_size=self.img_dimensions,
                                                                    seed=self.seed,
                                                                    shuffle=shuffle,
                                                                    batch_size=self.batch_size,
                                                                    validation_split=0.1,
                                                                    subset='validation'))
            self.class_labels = self.training_image_set[0].class_names
            for batch, _ in self.training_image_set[0]:
                self.img_count += batch[0]
            for batch, _ in self.validation_image_set[0]:
                self.img_count += batch[0]
            self.training_image_set[0] = self.training_image_set[0].cache().prefetch(buffer_size=tf.data.AUTOTUNE)
            self.validation_image_set[0] = self.validation_image_set[0].cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    def load_testing_images(self, testing_image_folder: str, image_size: int, color_mode: ColorMode = ColorMode.rgb):
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
