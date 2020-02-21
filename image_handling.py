import enum
import os
import cv2
import random
import numpy as np


class ColorMode(enum.Enum):
    RGB = 1
    BW = 2


class LabeledImages:
    def __init__(self, directory, folders, color, seed: int):
        # todo: would it make more sense if images were tuples of features and labels?
        self.features = []
        self.labels = []

        self.directory = directory  # todo: check if valid
        self.folders = folders
        self.color_mode = ColorMode.RGB if color else ColorMode.BW
        random.seed(seed)

    def load_images(self):
        for (index, folder_name) in enumerate(self.folders):
            self.load_from_filesystem(self.directory, folder_name, index, self.color_mode)
        self.randomize_order()

    def load_from_filesystem(self, directory, folder_name, class_num, color_mode):
        image_folder_path = os.path.join(directory, folder_name)
        for img in os.listdir(image_folder_path):
            img_array = cv2.imread(os.path.join(image_folder_path, img),
                                   cv2.IMREAD_COLOR if color_mode == ColorMode.RGB
                                   else cv2.IMREAD_GRAYSCALE)
            img_array = img_array / 255
            self.features.append(img_array)
            self.labels.append(class_num)
        print('Loaded images from %s.' % folder_name)

    def randomize_order(self):
        index = list(range(len(self.features)))
        random.shuffle(index)

        shuffled_image_features = [self.features[i] for i in index]
        shuffled_labels = [self.labels[i] for i in index]
        self.features = np.array(shuffled_image_features)
        self.labels = np.array(shuffled_labels)

        print('Shuffled image order.')
