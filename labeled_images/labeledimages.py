from labeled_images.colormode import ColorMode
import tensorflow as tf


class LabeledImages:
    def __init__(self, random_seed: int):
        self.seed: int = random_seed
        self.training_image_set: tf.data.Dataset = tf.data.Dataset.from_tensor_slices([0])
        self.validation_image_set: tf.data.Dataset = tf.data.Dataset.from_tensor_slices([0])
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

    def load_images_from_folders(self, training_images_location: str, image_size: int,
                                 color_mode: ColorMode = ColorMode.rgb, shuffle=True, n_folds=1, batch_size=32) -> None:
        self.color_mode = color_mode
        self.img_dimensions = (image_size, image_size)
        self.n_folds = n_folds  # TODO: implement splitting by folds
        self.batch_size = batch_size
        self.training_image_set = tf.keras.preprocessing.image_dataset_from_directory(training_images_location,
                                                                                      color_mode=self.color_mode.name,
                                                                                      image_size=self.img_dimensions,
                                                                                      seed=self.seed,
                                                                                      shuffle=shuffle,
                                                                                      batch_size=self.batch_size,
                                                                                      validation_split=0.1,
                                                                                      subset='training')
        self.validation_image_set = tf.keras.preprocessing.image_dataset_from_directory(training_images_location,
                                                                                        color_mode=self.color_mode.name,
                                                                                        image_size=self.img_dimensions,
                                                                                        seed=self.seed,
                                                                                        shuffle=shuffle,
                                                                                        batch_size=self.batch_size,
                                                                                        validation_split=0.1,
                                                                                        subset='validation')
        self.class_labels = self.training_image_set.class_names
        for batch, _ in self.training_image_set:
            self.img_count += batch[0]
        for batch, _ in self.validation_image_set:
            self.img_count += batch[0]
        self.training_image_set = self.training_image_set.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        self.validation_image_set = self.validation_image_set.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        # todo : confirm that the images are being normalized downstream

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
        # todo: remove test_features if nothing blows up
        self.test_features = list()
        for feature_class_pair in self.test_image_set.as_numpy_iterator():
            self.test_features = self.test_features + list(feature_class_pair[0])
            self.test_labels = self.test_labels + list(feature_class_pair[1])
