import os
import random
import argparse
from shutil import copyfile

SEED = 1


def parse_arguments():
    parser = argparse.ArgumentParser('import images and train model')
    parser.add_argument('-d', '--directory', default='', help='Folder holding category folders')
    parser.add_argument('-c1', '--category1', help='Folder of class 1')
    parser.add_argument('-c2', '--category2', help='Folder of class 2')
    parser.add_argument('-n', '--nsamples', help='Number of sample images')
    parser.add_argument('-g', '--groupsize', help='Group samples into sets of this size')
    args = parser.parse_args()

    img_directory = args.directory
    folders = [args.category1, args.category2]
    n_samples = int(args.nsamples)
    size_sample_group = int(args.groupsize)

    return img_directory, folders, n_samples, size_sample_group


if __name__ == '__main__':
    random.seed(SEED)
    img_directory, folders, n_samples, size_sample_group = parse_arguments()
    image_list = {}
    for f in folders:
        image_folder_path = os.path.join(img_directory, f)
        for img in os.listdir(image_folder_path):
            image_list[img] = f
    selected = random.sample(image_list.items(), k=n_samples)

    dest_folder = 'sample_of_' + str(size_sample_group) + '_images'
    for i in range(0, n_samples // size_sample_group):
        os.mkdir(dest_folder + str(i))
    for idx, (img, src_folder) in enumerate(selected):
        src = os.path.join(src_folder, img)
        dest = os.path.join(dest_folder + str(idx % n_samples // size_sample_group), src_folder + '_' + img)
        copyfile(src, dest)

    print(str(n_samples) + ' images selected and saved to ' + dest_folder + ' s.')
