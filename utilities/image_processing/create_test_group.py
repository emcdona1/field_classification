import os
import argparse
import shutil
import time

SPLIT = 10  # 1 image in every SPLIT is reserved for testing
VERBOSE = False


def process_input_arguments():
    ''' Parse command line arguments
    PARAMETERS:
    -----
    none (reads arguments from the command line)

    OUTPUT:
    -----
    @directory - file folder in working directory containing the image folders

    @categories - list of size 2, containing the image folder names (separated by species)

    @image_groups - dictionary with 4 keys, which contain the new folder names

    @verbose - how much detail to print during image copying process
    '''
    parser = argparse.ArgumentParser('data to be imported')
    parser.add_argument('-d', '--directory', default='', help='Folder holding category folders')
    parser.add_argument('-c1', '--category1', default='', help='1st folder')
    parser.add_argument('-c2', '--category2', default='', help='2nd folder')
    parser.add_argument('-c3', '--category3', default='', help='3rd folder')
    parser.add_argument('-c4', '--category4', default='', help='4th folder')
    parser.add_argument('-v', '--verbose', default='0', help='Verbose mode on (1) or off (0)')
    args = parser.parse_args()
    categories = [args.category1.replace('\\', '').replace('/', '').replace('.', ''),
                  args.category2.replace('\\', '').replace('/', '').replace('.', ''),
                  args.category3.replace('\\', '').replace('/', '').replace('.', ''),
                  args.category4.replace('\\', '').replace('/', '').replace('.', '')
                  ]
    image_groups = {categories[0] + 'train': [], categories[0] + 'test': [],
                    categories[1] + 'train': [], categories[1] + 'test': [],
                    categories[2] + 'train': [], categories[2] + 'test': [],
                    categories[3] + 'train': [], categories[3] + 'test': []
                    }
    global VERBOSE
    VERBOSE = (True if args.verbose == '1' else False)
    return args.directory, categories, image_groups


def split_images(directory, categories, image_groups):
    """ Split the images evenly into training and test sets

   PARAMETERS:
   -----
   @directory - directory containing the images

   @categories - the two image folder names inputted as arguments

   @image_groups - a dictionary containing four empty lists, with keys '[cat1]train' '[cat1]test', '[cat2]train' and '[cat2]test'

   OUTPUT:
   -----
   @image_groups - the same dictionary, with lists now filled with string file names

   """
    for category in categories:
        path = os.path.join(directory, category)
        for (idx, img_name) in enumerate(os.listdir(path)):
            if (idx % SPLIT) == SPLIT - 1:
                image_groups[category + 'test'].append(img_name)
            else:
                image_groups[category + 'train'].append(img_name)

    print('Category 0 split into ' + str(len(image_groups[categories[0] + 'train'])) + ' training images and ' + \
          str(len(image_groups[categories[0] + 'test'])) + ' testing images.')
    print('Category 1 split into ' + str(len(image_groups[categories[1] + 'train'])) + ' training images and ' + \
          str(len(image_groups[categories[1] + 'test'])) + ' testing images.\n')
    print('Category 2 split into ' + str(len(image_groups[categories[2] + 'train'])) + ' training images and ' + \
          str(len(image_groups[categories[2] + 'test'])) + ' testing images.')
    print('Category 3 split into ' + str(len(image_groups[categories[3] + 'train'])) + ' training images and ' + \
          str(len(image_groups[categories[3] + 'test'])) + ' testing images.\n')

    return image_groups


def copy_images_to_new_directories(directory, categories, image_groups):
    ''' Copies the images into new 'test' and 'train' folders.
    PARAMETERS:
    -----
    @directory - directory containing the images

    @categories - the two image folder names inputted as arguments

    @image_groups - a dictionary containing four empty lists, with keys '[cat1]train' '[cat1]test', '[cat2]train' and '[cat2]test'

    OUTPUT:
    -----
    nothing in Python (copies of images are saved in filesystem)
    '''
    # create new images folders
    for folder in image_groups:
        if not os.path.exists(os.path.join(directory, folder)):
            os.mkdir(os.path.join(directory, folder))
    for (group, img_list) in image_groups.items():
        count = 0
        for img in img_list:
            current_loc = os.path.join(directory, group.replace('test', '').replace('train', ''), img)
            new_loc = os.path.join(directory, group, img)
            shutil.copyfile(current_loc, new_loc)
            if VERBOSE:
                print(img + ' copied')
            else:
                print('.', end='', flush=True)
        if not VERBOSE:
            print()
        print(group + ' batch: ' + str(len(img_list)) + ' in ' + directory)


if __name__ == '__main__':
    # Start execution and parse arguments
    start_time = time.time()

    directory, categories, image_groups = process_input_arguments()
    image_groups = split_images(directory, categories, image_groups)
    copy_images_to_new_directories(directory, categories, image_groups)

    # Finish execution
    end_time = time.time()
    print('Task completed in %.1f seconds' % (end_time - start_time))
