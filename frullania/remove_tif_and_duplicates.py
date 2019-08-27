import os
import argparse

def remove(folder_path):
    cur_dir = os.listdir(folder_path)
    for img in cur_dir:
        if img[-4:] == '.tif':
            os.remove(os.path.join(folder_path,img))
        elif img[-6:] == '-0.jpg':
            os.remove(os.path.join(folder_path,img))
        elif img[-6:] == '-1.jpg':
            os.rename(os.path.join(folder_path,img), os.path.join(folder_path,img.replace('-1','')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('after using ImageMagick to convert .tif to .jpg, use this code to remove extraneous images')
    parser.add_argument('-f', '--folder', default='', help='Folder containing images with files to get rid of.')