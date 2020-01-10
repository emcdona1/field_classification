import enum
import os
import cv2
import random
import numpy as np


class ColorMode(enum.Enum):
    RGB = 1
    BW = 2


class ImageImporter:
    def __init__(self, img_directory, folders, img_size, import_in_color, seed):
        self.directory = img_directory
        self.folders = folders
        self.img_size = img_size
        self.color_mode = ColorMode.RGB if import_in_color else ColorMode.BW

        random.seed(seed)
        self.features = []
        self.labels = []
        self.load_from_filesystem()
        self.randomize_order()

    def load_from_filesystem(self):
        for (index, folder_name) in enumerate(self.folders):
            image_folder_path = os.path.join(self.directory, folder_name)
            for img in os.listdir(image_folder_path):
                img_array = cv2.imread(os.path.join(image_folder_path, img),
                                       cv2.IMREAD_COLOR if self.color_mode == ColorMode.RGB
                                       else cv2.IMREAD_GRAYSCALE)
                img_array = img_array / 255

                self.features.append(img_array)
                self.labels.append(index)
        print('Loaded images.')

    def randomize_order(self):
        index = list(range(len(self.features)))
        random.shuffle(index)

        shuffled_features = [self.features[i] for i in index]
        shuffled_labels = [self.labels[i] for i in index]
        self.features = np.array(shuffled_features)
        self.labels = np.array(shuffled_labels)

        print('Shuffled data.')
