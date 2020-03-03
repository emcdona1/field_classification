import os
from labeled_images.colormode import ColorMode
import cv2
import random
import numpy as np


class LabeledImages:
    def __init__(self, folders, color, seed: int):
        # todo: would it make more sense if images were tuples of features and labels?
        self.features = []
        self.labels = []

        self.folders = folders
        self.color_mode = ColorMode.RGB if color else ColorMode.BW
        random.seed(seed)

        self.load_images()

    def load_images(self):
        for (index, folder_name) in enumerate(self.folders):
            self.load_from_filesystem(folder_name, index)
        self.randomize_order()

    def load_from_filesystem(self, image_folder_path, class_num):
        for img in os.listdir(image_folder_path):
            img_array = cv2.imread(os.path.join(image_folder_path, img),
                                   cv2.IMREAD_COLOR if self.color_mode == ColorMode.RGB
                                   else cv2.IMREAD_GRAYSCALE)
            img_array = img_array / 255
            self.features.append(img_array)
            self.labels.append(class_num)
        print('Loaded images from %s.' % image_folder_path)

    def randomize_order(self):
        index = list(range(len(self.features)))
        random.shuffle(index)

        shuffled_image_features = [self.features[i] for i in index]
        shuffled_labels = [self.labels[i] for i in index]
        self.features = np.array(shuffled_image_features)
        self.labels = np.array(shuffled_labels)

        print('Shuffled image order.')

    def subset(self, index_list):
        subset_features = self.features[index_list]
        subset_labels = self.labels[index_list]
        return subset_features, subset_labels
