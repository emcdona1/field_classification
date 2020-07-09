import os
from labeled_images.colormode import ColorMode
import cv2
import random
import numpy as np
from tensorflow.keras import datasets


class LabeledImages:
    def __init__(self, seed: int):
        self.features = np.array((0, 0))
        self.labels = np.array((0, 0))
        self.img_names = None
        self.n_images = 0
        self.color_mode = None
        self.img_dim = 0
        self.class_labels = None

    def load_images_from_folders(self, folders, color_mode, class_labels) -> None:
        print('Loading image set from filesystem.')
        self.features = []
        self.labels = []
        self.img_names = []
        self.class_labels = class_labels
        self.color_mode = ColorMode.RGB if color_mode else ColorMode.BW
        print(self.color_mode == ColorMode.RGB)

        for (class_num, image_folder_path) in enumerate(folders):
            for img in os.listdir(image_folder_path):
                img_array = cv2.imread(os.path.join(image_folder_path, img),
                                       cv2.IMREAD_COLOR if self.color_mode == ColorMode.RGB
                                       else cv2.IMREAD_GRAYSCALE)
                img_array = img_array / 255
                img_array = img_array.reshape([img_array.shape[0], img_array.shape[1],
                                               3 if self.color_mode == ColorMode.RGB else 1])
                self.features.append(img_array)
                self.labels.append(class_num)
                self.img_names.append(img)
            print('Loaded images from %s.' % image_folder_path)
        self.randomize_order()
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        self.img_names = np.array(self.img_names)
        # self.features.shape = (# of images, img dimension, img dimension, color channels)
        self.n_images = self.features.shape[0]
        self.img_dim = self.features.shape[1]

    def load_cifar_images(self) -> None:
        print('Loading CIFAR-10 image set (28x28x3).')
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

        self.n_images = self.features.shape[0]
        self.color_mode = ColorMode.RGB
        self.img_dim = self.features[0].shape[1]

    def randomize_order(self) -> None:
        index = list(range(len(self.features)))
        random.shuffle(index)

        shuffled_image_features = [self.features[i] for i in index]
        shuffled_labels = [self.labels[i] for i in index]
        if self.img_names:
            shuffled_names = [self.img_names[i] for i in index]
            self.img_names = shuffled_names
        self.features = shuffled_image_features
        self.labels = shuffled_labels
        print('Shuffled image order.')

    def subset(self, index_list) -> (np.array, np.array):
        subset_features = self.features[index_list]
        subset_labels = self.labels[index_list]
        return subset_features, subset_labels
