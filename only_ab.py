import os
import sys
import cv2
import numpy as np


def main(list_of_folders: list):
    for folder in list_of_folders:
        files = os.listdir(folder)
        ab_graveyard_folder = os.path.join(folder, 'abaxial')
        os.makedirs(ab_graveyard_folder)
        abaxial = [f for f in files if '_abaxial_' in os.path.join(folder, f)]
        for ab in abaxial:
            ab_location = os.path.join(folder, ab)
            image = open_cv2_image(ab_location)
            jpg_filename = ab[:-4] + '.jpg'
            save_location = save_cv2_image(folder, jpg_filename, image)
            save_location = save_cv2_image(ab_graveyard_folder, ab, image)
            os.remove(ab_location)
            print('%s Saved abaxial file to.' % ab_location)


def open_cv2_image(image_location: str) -> np.ndarray:
    return cv2.imread(image_location)


def save_cv2_image(save_location: str, save_filename: str, image_to_save: np.ndarray) -> str:
    file_path = os.path.join(save_location, save_filename)
    cv2.imwrite(file_path, image_to_save)
    return file_path


if __name__ == '__main__':
    """ Given 1 or more image folders, saves any TIFs as JPGS and moves TIFs to a new subdirectory 'tifs.'"""
    assert len(sys.argv) >= 2, 'Please specify at least one image folder.'
    folders = sys.argv[1:]
    main(folders)