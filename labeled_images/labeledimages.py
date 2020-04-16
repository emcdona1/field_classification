import os
from labeled_images.colormode import ColorMode
import cv2
import random
import numpy as np
from tensorflow.keras import datasets


class LabeledImages:
    def __init__(self, seed: int):
        # todo: would it make more sense if images were tuples of features and labels?
        self.features = np.array((0, 0))
        self.labels = np.array((0, 0))
        self.n_images = 0
        self.color_mode = None
        self.img_dim = 0
        self.class_labels = None

    def load_images_from_folders(self, folders, color, class_labels) -> None:
        self.features = []
        self.labels = []
        self.class_labels = class_labels
        self.color_mode = ColorMode.RGB if color else ColorMode.BW

        for (class_num, image_folder_path) in enumerate(folders):
            for img in os.listdir(image_folder_path):
                img_array = cv2.imread(os.path.join(image_folder_path, img),
                                       cv2.IMREAD_COLOR if self.color_mode == ColorMode.RGB
                                       else cv2.IMREAD_GRAYSCALE)
                img_array = img_array / 255
                self.features.append(img_array)
                self.labels.append(class_num)
            print('Loaded images from %s.' % image_folder_path)
        self.randomize_order()
        self.n_images = self.features[0].shape[0]
        self.img_dim = self.features[0].shape[1]

    def load_cifar_images(self) -> None:
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

        self.randomize_order()
        self.n_images = self.features.shape[0]
        self.color_mode = ColorMode.RGB
        self.img_dim = self.features[0].shape[1]

    def randomize_order(self) -> None:
        index = list(range(len(self.features)))
        random.shuffle(index)

        shuffled_image_features = [self.features[i] for i in index]
        shuffled_labels = [self.labels[i] for i in index]
        self.features = np.array(shuffled_image_features)
        self.labels = np.array(shuffled_labels)
        print('Shuffled image order.')

    def subset(self, index_list) -> (np.array, np.array):
        subset_features = self.features[index_list]
        subset_labels = self.labels[index_list]
        return subset_features, subset_labels

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
