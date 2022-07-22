import os
import sys
from pathlib import Path

import shutil
import datetime
import pandas as pd

from timer import Timer
from dataloader import get_timestamp_for_file_saving


def main(images_folder, voted_csv):
    timer = Timer('Sort voted images')
    assert os.path.isdir(images_folder), f'Invalid 1st argument: {images_folder} is not a file.'
    assert os.path.isfile(voted_csv), f'Invalid 1st argument: {voted_csv} is not a file.'
    voted_data = pd.read_csv(voted_csv, dtype={'image_file':str})

    sorted_images_dir = f'utilities/sorted_images/{images_folder.name}/{get_timestamp_for_file_saving()}/'
    sorted_images_subfolders = dict.fromkeys(list(voted_data.voted_sex.unique()))
    for folder in list(voted_data.voted_sex.unique()):
        print(folder)
        dir = Path(sorted_images_dir + folder)
        sorted_images_subfolders[folder] = dir
        if not os.path.exists(dir):
            os.makedirs(dir)
    print(sorted_images_subfolders)
    print(voted_data['image_file'])
    for root, dirs, files in os.walk(images_folder):
        for _, row in voted_data.iterrows():
            image_name = row['image_file']
            sex_vote = row['voted_sex']
            if image_name in files:
                path_to = os.path.join(sorted_images_subfolders[sex_vote], image_name)
                try:
                    path_from = os.path.join(root, image_name)
                    shutil.move(path_from, path_to)
                except FileNotFoundError:
                    pass
                for dir in dirs:
                    try:
                        path_from = os.path.join(root, dir, image_name)
                        shutil.move(path_from, path_to)
                    except FileNotFoundError:
                        pass


if __name__ == '__main__':
    images_directory = Path(sys.argv[1])
    voted_csv = Path(sys.argv[2])
    main(images_directory, voted_csv)
