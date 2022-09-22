import os
import cv2
import numpy as np
import sys
from pathlib import Path
from typing import List


def main(base_folder: Path):
    image_folders = [Path(base_folder, str(f)) for f in os.listdir(base_folder)]
    image_names = [os.listdir(f) for f in image_folders]
    max_height, max_width = find_new_dimensions(image_folders, image_names)

    base_save_folder = Path(f'{base_folder}_padded')
    save_folders = [Path(base_save_folder, str(f)) for f in os.listdir(base_folder)]
    for folder in save_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    for idx, directory in enumerate(image_folders):
        for image in image_names[idx]:
            current_image = cv2.imread(os.path.join(directory, image))
            height, width, channels = current_image.shape
            fill_color = (255, 255, 255)
            result = np.full((max_height, max_width, channels), fill_color, dtype=np.uint8)
            # Find upper left point for placing the original image, and copy to center of canvas
            xx = (max_width - width) // 2
            yy = (max_height - height) // 2
            result[yy:yy + height, xx:xx + width] = current_image
            cv2.imwrite(os.path.join(save_folders[idx], image), result)
    print(f'Padded images saved to: {base_save_folder}.')


def find_new_dimensions(image_folders: List[Path], image_names: List[List[str]]):
    max_height = 0
    max_width = 0
    for idx, directory in enumerate(image_folders):
        for image in image_names[idx]:
            current_image = cv2.imread(os.path.join(directory, image))
            height, width, _ = current_image.shape
            max_height = max(height, max_height)
            max_width = max(width, max_width)
    print(f'Max height: {max_height}, max width: {max_width}')
    return max_height, max_width


if __name__ == '__main__':
    assert len(sys.argv) == 2, 'Please provide one argument of a folder with images ' +\
                               '(folder containing one subfolder for each class).'
    folder = Path(sys.argv[1])
    assert folder.is_dir(), f'Not a valid directory: {folder.absolute()}'
    main(folder)
