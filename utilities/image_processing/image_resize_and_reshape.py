import argparse
import cv2
import os
import numpy as np


def parse_arguments() -> (str, str, int):
    parser = argparse.ArgumentParser('Images to resize')
    parser.add_argument('-source', '--source_directory', help='Path to folder of original images')
    parser.add_argument('-dest', '--destination_directory', default='', help='Path to save resized images')
    parser.add_argument('-size', '--image_size', default=256, help='The new image height & width (output is square)')
    args = parser.parse_args()
    return args.source, args.destination, int(args.image_size)


def reduce_image_dim(filename: str) -> np.ndarray:
    img: np.ndarray = cv2.imread(os.path.join(source_folder, filename), -1)
    try:
        img_rsz: np.ndarray = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    except cv2.error:
        print('%s resize failed.' % filename)
        img_rsz = np.ndarray((0, 0))
    return img_rsz


def save_image(filename: str, img: np.ndarray) -> None:
    try:
        cv2.imwrite(os.path.join(destination_folder, filename), img)
    except cv2.error:
        print('`%s` did not save successfully.' % img)


if __name__ == '__main__':
    source_folder, destination_folder, img_size = parse_arguments()
    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)

    for (idx, img_filename) in enumerate(os.listdir(source_folder)):
        resized_image = reduce_image_dim(img_filename)
        save_image(img_filename, resized_image)
        if idx % 500 == 499:
            print('500 images processed.')
    print('Images resizing complete.')
