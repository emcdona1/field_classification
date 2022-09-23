import os
import sys
import shutil
from pathlib import Path
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import cv2
import random
import numpy as np


SEED = 1


def main(image_source_folder: Path):
    base_save_folder = Path(f'{str(image_source_folder)}_augmented')
    save_folders = [Path(base_save_folder, str(f)) for f in os.listdir(image_source_folder)]
    if not os.path.exists(base_save_folder):
        os.makedirs(base_save_folder)
    for folder in save_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    training_images = tf.keras.preprocessing.image_dataset_from_directory(image_source_folder, batch_size=32)

    data_augmentation_0 = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical', seed=SEED),
        tf.keras.layers.experimental.preprocessing.RandomRotation((-0.25, 0.25), seed=SEED)
        # rotation is randomly between +/-0.25 * 2pi, e.g. this is +/- 90°
    ])
    data_augmentation_1 = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomRotation((-1/12, 1/12), seed=SEED),
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical', seed=SEED),
    ])
    data_augmentation_2 = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomRotation((-1/12, 1/12), seed=SEED),
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical', seed=SEED),
        tf.keras.layers.experimental.preprocessing.RandomRotation((-0.5, -0.5), seed=SEED),
    ])
    data_augmentation_3 = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('vertical', seed=SEED),
        tf.keras.layers.experimental.preprocessing.RandomRotation((-0.5, 0.5), seed=SEED),
        tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=(-0.1, 0.1),
                                                              width_factor=(-0.1, 0.1), seed=SEED)
    ])

    process_augmentation(base_save_folder, training_images, data_augmentation_0)
    process_augmentation(base_save_folder, training_images, data_augmentation_3)
    copy_original_images(image_source_folder, save_folders)


def process_augmentation(base_save_folder, training_images, data_augmentation):
    for batch_images, batch_labels in training_images.as_numpy_iterator():
        augmented_images = data_augmentation(batch_images)
        for idx in range(augmented_images.shape[0]):
            image = np.array(augmented_images[idx])
            label = batch_labels[idx]
            save_loc = Path(base_save_folder, training_images.class_names[label],
                            f'{random.randint(10000, 999999)}.jpg')
            cv2.imwrite(str(save_loc), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def copy_original_images(image_source_folder, save_folders):
    image_folders = [Path(image_source_folder, str(f)) for f in os.listdir(image_source_folder)]
    image_names = [(str(s) for s in os.listdir(f)) for f in image_folders]
    for idx, folder in enumerate(image_folders):
        for img in image_names[idx]:
            shutil.copy2(Path(folder, img), Path(save_folders[idx], img))


if __name__ == '__main__':
    assert len(sys.argv) == 2, 'Please specify 1 argument: 1) a folder that contains folders of TRAINING images.  ' +\
                               '(Note: you should not augment testing images).'
    training_image_folder = Path(sys.argv[1])
    assert training_image_folder.exists() and training_image_folder.is_dir(), \
        f'Not a valid folder path: {training_image_folder.absolute()}'
    batch_size_1 = 64

    main(training_image_folder)
