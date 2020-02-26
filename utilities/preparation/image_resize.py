import argparse
import cv2
import os

def parse_arguments():
    parser = argparse.ArgumentParser('Images to resize')
    parser.add_argument('-sf', '--source', help='Path to folder of original images')
    parser.add_argument('-df', '--destination', default='', help='Path to save resized images')
    parser.add_argument('-s', '--image_size', default=256, help='The new image height & width (output is square)')
    args = parser.parse_args()
    return args.source, args.destination, int(args.image_size)

def reduce_image_dim(img_filename, source, img_size):
    img = cv2.imread(os.path.join(source, img_filename), -1)
    try:
        img_rsz = cv2.resize(img, (img_size, img_size), interpolation = cv2.INTER_AREA)
    except cv2.error as ce:
        print(img_filename + ' resize failed.')
    return img_rsz

def save_image(img_filename, resized_image, destination):
    try:
        cv2.imwrite(os.path.join(destination, img_filename), resized_image)
    except cv2.error as ce:
        print(resized_image, ' did not save.')

if __name__== '__main__':
    source, destination, img_size = parse_arguments()
    if not os.path.exists(destination):
        os.mkdir(destination)
    
    for (idx, img_filename) in enumerate(os.listdir(source)):
        resized_image = reduce_image_dim(img_filename, source, img_size)
        save_image(img_filename, resized_image, destination)
        if idx % 500 == 499:
            print('500 images processed.')