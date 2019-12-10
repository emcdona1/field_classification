import enum
import os
import cv2
import random
import numpy as np


class ColorMode(enum.Enum):
    RGB = 1
    BW = 2


class ImageLoader:
    def __init__(self, img_directory, folders, img_size, import_in_color, seed):
        self.directory = img_directory
        self.folders = folders
        self.img_size = img_size
        self.color_mode = ColorMode.RGB if import_in_color else ColorMode.BW

        random.seed(seed)
        self.features = []
        self.labels = []
        self.load_images()

    def load_images(self):
        all_data = self.load_from_filesystem()
        self.separate_information(all_data)

    def load_from_filesystem(self):
        all_data = []
        for (class_index, category) in enumerate(self.folders):
            image_folder_path = os.path.join(self.directory, category)
            for img in os.listdir(image_folder_path):
                img_array = cv2.imread(os.path.join(image_folder_path, img),
                                       cv2.IMREAD_COLOR if self.color_mode == ColorMode.RGB
                                       else cv2.IMREAD_GRAYSCALE)
                img_array = img_array / 255.0
                all_data.append([img_array, class_index])
        random.shuffle(all_data)
        print("Loaded and shuffled data.")
        return all_data

    def separate_information(self, all_data):
        # store the image features (array of RGB for each pixel) and labels into corresponding arrays
        for data_feature, data_label in all_data:
            self.features.append(data_feature)
            self.labels.append(data_label)

        # reshape into numpy array
        self.features = np.array(self.features)
        self.features = self.features.reshape(-1, self.img_size, self.img_size,
                                              3 if self.color_mode == ColorMode.RGB else 1)
        self.labels = np.array(self.labels)
        print("Stored features and labels.")