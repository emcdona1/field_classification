import enum
import os
import cv2
import random
import numpy as np


# todo: use this one, delete image_set asap

class ColorMode(enum.Enum):
    RGB = 1
    BW = 2


def import_images(directory, folders, img_size, import_in_color, seed):
    color_mode = ColorMode.RGB if import_in_color else ColorMode.BW
    random.seed(seed)
    imgset = LabeledImages()
    for (index, folder_name) in enumerate(folders):
        imgset.load_from_filesystem(directory, folder_name, index, color_mode)

    imgset.randomize_order()


class LabeledImages:
    def __init__(self):
        self.features = []
        self.labels = []

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
