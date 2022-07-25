import os
import sys
import cv2
import numpy as np


def main(list_of_folders: list):
    for folder in list_of_folders:
        # print(folder)
        files = os.listdir(folder)
        # print(files)
        imgs = [f for f in files if ' ' in os.path.join(folder, f)]
        for img in imgs:
            # print(img)
            # print(os.path.join(folder, img))
            img = os.path.join(folder, img)
            os.rename(img, img.replace(' ', ''))
            print(f'Replaced {img}')
        # for img in os.listdir(folder):
        #     print(img)
        #     os.rename(img, img.replace(' ', ''))



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