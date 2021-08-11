import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
from labeled_images import LabeledImages
from models import TransferLearningModel
from models.model_training import ModelTrainer
from utilities.timer import Timer
from models.modeltrainingarguments import ModelTrainingArguments
import matplotlib
import random

matplotlib.use('Agg')  # required when running on server


def main() -> None:
    timer = Timer('Model training')

    model_trainer, images = set_up_components('base_model_location')

    model_trainer.train_and_save_all_models(images)
    model_trainer.charts.finalize()

    print('class 1: ' + images.class_labels[0] + ', class 2: ' + images.class_labels[1])
    timer.stop()
    timer.print_results()


def set_up_components(base_model_location: str) -> (ModelTrainer, LabeledImages):
    cnn_arguments = ModelTrainingArguments()
    images = LabeledImages(SEED)
    images.load_training_images(cnn_arguments.training_image_folder, cnn_arguments.image_size,
                                cnn_arguments.color_mode, True, cnn_arguments.n_folds, cnn_arguments.batch_size)
    base_model = tf.keras.models.load_model(base_model_location)
    architecture = TransferLearningModel(base_model, SEED, cnn_arguments.lr,
                                         images.img_dimensions[0], images.color_mode)
    model_trainer = ModelTrainer(cnn_arguments.n_epochs, cnn_arguments.batch_size, cnn_arguments.n_folds,
                                 architecture, SEED)
    return model_trainer, images


def train_base_model():
    num_classes = 47
    image_dimensions = (64, 64, 3)
    image_size = image_dimensions[0:2]

    def sequential_api_layers() -> tf.keras.Model:
        # preprocessing layers
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=image_dimensions))
        model.add(keras.Sequential([
            keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
            keras.layers.experimental.preprocessing.RandomRotation(0.2),
        ]))
        model.add(keras.layers.experimental.preprocessing.Rescaling(1 / 255, input_shape=image_dimensions))

        # convolutional layers
        model.add(keras.layers.Conv2D(10, (5, 5), input_shape=(image_dimensions[0], image_dimensions[1], 3)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(keras.layers.Conv2D(40, (5, 5)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dropout(0.2))

        # dense layers/output
        model.add(keras.layers.Dense(500, activation='linear'))
        model.add(keras.layers.Dense(500, activation='relu'))
        model.add(keras.layers.Dense(num_classes))

        return model

    def functional_api_layers() -> tf.keras.Model:
        # preprocessing layers
        input = keras.Input(shape=image_dimensions)
        x = keras.layers.experimental.preprocessing.RandomFlip('horizontal')(input)
        x = keras.layers.experimental.preprocessing.RandomRotation(0.2)(x)
        # normalize the inputs from [0,255] to [-1, 1]
        # Normalization calculates as outputs = (inputs - mean) / sqrt(var)
        # norm_layer = keras.layers.experimental.preprocessing.Normalization()
        # mean = np.array([255 / 2] * 3)
        # var = mean ** 2
        # x = norm_layer(x)
        # norm_layer.set_weights([mean, var])
        x = keras.layers.experimental.preprocessing.Rescaling(1 / 255, input_shape=image_dimensions)(x)

        # # convolutional layers
        x = keras.layers.Conv2D(10, (5, 5), input_shape=(image_dimensions[0], image_dimensions[1], 3))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = keras.layers.Conv2D(40, (5, 5))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

        x = keras.layers.Flatten()(x)
        x = keras.layers.Dropout(0.2)(x)

        # # dense layers/output
        x = keras.layers.Dense(500, activation='linear')(x)
        x = keras.layers.Dense(500, activation='relu')(x)
        output = keras.layers.Dense(num_classes)(x)

        model = keras.Model(input, output)
        return model

    model = functional_api_layers()

    # load images
    tfds_random_seed = tfds.ReadConfig(shuffle_seed=SEED)
    (train_ds, validation_ds, test_ds), ds_info = tfds.load(
        'emnist/bymerge',
        split=['train[:40%]', 'train[40%:50%]', 'test'],
        as_supervised=True, with_info=True,
        read_config=tfds_random_seed,
        shuffle_files=True
    )
    # resize TFDS images
    train_ds = train_ds.map(lambda img, label: (tf.image.resize(img, image_size), label))
    validation_ds = validation_ds.map(lambda img, label: (tf.image.resize(img, image_size), label))
    test_ds = test_ds.map(lambda img, label: (tf.image.resize(img, image_size), label))

    # use prefetch to optimize loading speed
    batch_size = 32
    train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=10)
    validation_ds = validation_ds.cache().batch(batch_size).prefetch(buffer_size=10)
    test_ds = test_ds.cache().batch(batch_size).prefetch(buffer_size=10)

    # Convert grayscale images to RGB
    train_ds = train_ds.map(lambda img, label: (tf.image.grayscale_to_rgb(img), label))
    validation_ds = validation_ds.map(lambda img, label: (tf.image.grayscale_to_rgb(img), label))
    test_ds = test_ds.map(lambda img, label: (tf.image.grayscale_to_rgb(img), label))
    label_to_letter = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
                       10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
                       20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
                       30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'd', 39: 'e',
                       40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'}
    # reused: c, i, j, k, l, m, o, p, s, u, v, w, x, y, z
    letter_to_label = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                       'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19,
                       'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29,
                       'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35, 'a': 36, 'b': 37, 'c': 12, 'd': 38,
                       'e': 39, 'f': 40, 'g': 41, 'h': 42, 'i': 18, 'j': 19, 'k': 20, 'l': 21, 'm': 22, 'n': 43,
                       'o': 24, 'p': 25, 'q': 44, 'r': 45, 's': 28, 't': 46, 'u': 30, 'v': 31, 'w': 32, 'x': 33,
                       'y': 34, 'z': 35}

    # compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=0.00001, decay=0.01,
                                        amsgrad=False),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', keras.metrics.SparseTopKCategoricalAccuracy(k=5)])

    # train and save model
    epochs = 20
    model.fit(train_ds, epochs=epochs, validation_data=validation_ds, verbose=2, batch_size=64)
    local_computer_save_location = os.path.join('file_resources', 'mnistmerge-20e.model')
    google_colab_save_location = 'drive/MyDrive/mnistmerge-20e.model'
    model.save(local_computer_save_location)


if __name__ == '__main__':
    # set up random seeds
    SEED = 1
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)
    main()
