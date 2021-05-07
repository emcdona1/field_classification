import codecs
import os
from labeled_images.colormode import ColorMode
import cv2
import numpy as np
from tensorflow.keras import datasets
from typing import Tuple, List


class LabeledImages:
    """ Class to hold image sets for training and testing in a neural network. Images must be square."""
    def __init__(self, seed=None):
        self.features: np.ndarray = np.array([])
        self.labels: np.ndarray = np.array([])
        self.image_names: np.ndarray = np.array([])
        self.no_of_images: int = 0
        self.color_mode: ColorMode = ColorMode.RGB
        self.image_dimension: int = 0
        self.class_labels: List[str, ...] = ['', '']
        self.seed = seed

    def load_images_from_folders(self, folders: Tuple[str, str], color_mode: ColorMode, class_labels: List[str, ...]) \
            -> None:
        """ Given a pair of folders - one for each class, using a specified color mode (RGB or BW), and a pair
        of class labels, load the images and labels from the filesystem into memory."""
        features = []
        labels = []
        image_names = []
        self.class_labels: List[str, ...] = class_labels
        self.color_mode: ColorMode = color_mode

        for (class_num, image_folder_path) in enumerate(folders):
            for image in os.listdir(image_folder_path):
                image_features = self.load_cv2_image(os.path.join(image_folder_path, image))
                features.append(image_features)
                labels.append(class_num)
                image_names.append(image)
            print('Loaded "%s" class images from: %s' % (self.class_labels[class_num], image_folder_path))
        self.no_of_images = len(features)
        self.randomize_order()
        self.features = np.array(features)
        print('Image collection shape: ' + str(self.features.shape))
        self.labels = np.array(labels)
        self.image_names = np.array(image_names)
        # self.features.shape = (# of images, image dimension, image dimension, color channels)
        print(self.features.shape)
        self.image_dimension = self.features.shape[1]

    def load_cv2_image(self, image_path):
        image_array = cv2.imread(image_path,
                               cv2.IMREAD_COLOR if self.color_mode == ColorMode.RGB
                               else cv2.IMREAD_GRAYSCALE)
        image_array = image_array / 255
        if self.color_mode == ColorMode.BW:
            image_array = image_array.reshape([image_array.shape[0], image_array.shape[1], 1])
        return image_array

    def load_from_csv(self, csv_filename: str, class_labels: (str, str)):
        pass

    def load_cifar_images(self) -> None:
        print('Loading 2 classes from the CIFAR-10 image set (28x28x3).')
        (train_features, train_labels), (val_features, val_labels) = datasets.cifar10.load_data()
        train_features, test_features = train_features / 255.0, val_features / 255.0
        # class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # reduce to just cat and dog images (2 classes)
        self.class_labels = ('cat', 'dog')
        cat_dog_train_mask = [True if (a == 3 or a == 5) else False for a in train_labels]
        cat_dog_test_mask = [True if (a == 3 or a == 5) else False for a in val_labels]

        train_features = np.array([a for (idx, a) in enumerate(train_features) if cat_dog_train_mask[idx]])
        val_features = np.array([a for (idx, a) in enumerate(test_features) if cat_dog_test_mask[idx]])
        self.features = np.vstack((train_features, val_features))

        train_labels = np.array([a // 4 for (idx, a) in enumerate(train_labels) if cat_dog_train_mask[idx]])
        val_labels = np.array([a // 4 for (idx, a) in enumerate(val_labels) if cat_dog_test_mask[idx]])
        self.labels = np.vstack((train_labels, val_labels))

        self.no_of_images = self.features.shape[0]
        self.color_mode = ColorMode.RGB
        self.image_dimension = self.features[0].shape[1]

    def randomize_order(self) -> None:
        seed = self.seed
        if not seed:
            print('Warning: no random seed set for LabeledImages, picking a random integer.')
            seed = np.random.randint(0,10000)
        np.random.seed(seed)
        np.random.shuffle(self.features)
        np.random.seed(seed)
        np.random.shuffle(self.labels)
        np.random.seed(seed)
        np.random.shuffle(self.image_names)
        print('Shuffled image order.')

    def subset(self, index_list) -> (np.array, np.array):
        subset_features = self.features[index_list]
        subset_labels = self.labels[index_list]
        return subset_features, subset_labels


class MulticlassLabeledImages(LabeledImages):
    def load_nist_letter_images(self, base_folder: str, color_mode=ColorMode.BW, seed=None) -> None:
        """ Given an arbitrary number of folders containing images for an arbitrary number of classes
        (named by class_labels), load the images and labels from the filesystem into memory.

        (optional) seed argument will set the random seed when shuffling images.  (If a seed has been loaded previously,
        that seed will be used.) """
        if seed:
            self.seed = seed
        features = []
        labels = []
        image_names = []
        self.color_mode: ColorMode = color_mode
        self.class_labels: List[str, ...] = self.get_class_names(base_folder)
        for dir_path, contained_dirs, contained_files in os.walk(base_folder):
            print('Currently loading: %s' % dir_path)
            for filename in contained_files:
                if '.png' in os.path.join(dir_path, filename):
                    image_features = self.load_cv2_image(os.path.join(dir_path, filename))
                    features.append(image_features)
                    image_class = self.parse_nist_image_class(dir_path)
                    labels.append(self.class_labels.index(image_class))
                    image_names.append(filename)
        self.no_of_images = len(features)
        self.features = np.array(features)
        print('Image collection shape: ' + str(self.features.shape))
        self.labels = np.array(labels)
        self.image_names = np.array(image_names)
        self.image_dimension = self.features.shape[1]

        if self.seed:
            self.randomize_order()
        print('Images loaded: %i (%i x %i images in %i-channel color)' % self.features.shape)

    def parse_nist_image_class(self, dir_path: str) -> str:
        image_class = dir_path.split(os.path.sep)[-2]
        image_class = image_class.split('_')[0]
        image_class = codecs.decode(image_class, 'hex')
        image_class = str(image_class, 'ascii')
        image_class = image_class.lower()
        return image_class

    def get_class_names(self, base_folder: str) -> list:
        _, classes, _ = os.walk(base_folder).__next__()
        classes_list = [c.split('_')[0] for c in classes]
        classes_list = [str(codecs.decode(c, 'hex'), 'ascii').lower() for c in classes_list]
        classes_set = set(classes_list)
        classes_list_consolidated = sorted(classes_set)
        return classes_list_consolidated
