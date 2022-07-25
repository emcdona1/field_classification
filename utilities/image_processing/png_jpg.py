import os
import sys
import cv2
import numpy as np


def main(list_of_folders: list):
    for folder in list_of_folders:
        files = os.listdir(folder)
        # png_graveyard_folder = os.path.join(folder, 'pngs')
        # os.makedirs(png_graveyard_folder)
        pngs = [f for f in files if '.png' in os.path.join(folder, f)]
        for png in pngs:
            png_location = os.path.join(folder, png)
            image = open_cv2_image(png_location)
            jpg_filename = png[:-4] + '.jpg'
            save_location = save_cv2_image(folder, jpg_filename, image)
            # save_location = save_cv2_image(png_graveyard_folder, png, image)
            os.remove(png_location)
            print('%s converted to jpg.' % png_location)


def open_cv2_image(image_location: str) -> np.ndarray:
    return cv2.imread(image_location)


def save_cv2_image(save_location: str, save_filename: str, image_to_save: np.ndarray) -> str:
    file_path = os.path.join(save_location, save_filename)
    cv2.imwrite(file_path, image_to_save)
    return file_path


if __name__ == '__main__':
    """ Given 1 or more image folders, saves any PNGs as JPGS and moves PNGs to a new subdirectory 'pngs.'"""
    assert len(sys.argv) >= 2, 'Please specify at least one image folder.'
    folders = sys.argv[1:]
    main(folders)