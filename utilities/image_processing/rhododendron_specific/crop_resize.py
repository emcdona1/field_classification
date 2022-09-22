import sys
from pathlib import Path
import numpy as np
import cv2
import csv


def pad_based_on_csv(csv_filename: Path):
    count = 0
    with open(csv_filename, 'r') as file:
        height, width = find_new_dimensions(file)

        reader = csv.reader(file)
        for row in reader:
            if row:
                path = str(row[2])
                img = cv2.imread(path)
                ht, wd, cc = img.shape

                # create new image of desired size and color for padding
                ww = width
                hh = height
                color = (255, 254, 252)
                result = np.full((hh, ww, cc), color, dtype=np.uint8)

                # Find center of canvas to place the original image
                xx = (ww - wd) // 2
                yy = (hh - ht) // 2

                # copy image into center of canvas
                result[yy:yy + ht, xx:xx + wd] = img
                count += 1

                # Split our filename to get the desired sections and exclude the .jpg or .png extension
                path_split = path.split("/")
                file = path_split[-1].replace('.jpg', '')
                file = file.replace('.png', '')

                # Merge back together our file name excluding the parts we do not want
                path = path_split[:-1]
                path = '/'.join(path)
                print(path)
                cv2.imwrite(path + '/' + "Autopadded_" + file + ".jpg", result)
                print(f'Saved {count}')


def pad_based_on_folder(folder: Path):
    pass


def find_new_dimensions(file):
    # Function to find the largest dimensions stored in the CSV file
    max_height = 0
    max_width = 0

    reader = csv.reader(file)
    for row in reader:
        # Ensure row is not null
        if row:
            height = int(row[0])
            width = int(row[1])
            if height > max_height:
                max_height = height
            if width > max_width:
                max_width = width

    return max_height, max_width


if __name__ == '__main__':
    """ Given either a CSV or a folder of images, pad all images with whitespace, so that they're all the same 
    size/dimensions in pixels."""
    if len(sys.argv) == 1:  #todo: remove this
        sys.argv[1] = 'auto_crop_image_data.csv'
    assert len(sys.argv) == 2, 'Please provide 1 argument (either a CSV file, or a folder).'
    argument = Path(sys.argv[1])
    if 'csv' in argument.suffix:
        pad_based_on_csv(argument)
    else:
        pad_based_on_folder(argument)
