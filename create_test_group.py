import os
import argparse
import shutil
import time

SPLIT = 10 # 1 image in every SPLIT is reserved for testing

def process_input_arguments():
    parser = argparse.ArgumentParser('data to be imported')
    parser.add_argument('-d', '--directory', default='', help='Folder holding category folders')
    parser.add_argument('-c1', '--category1', default='', help='1st folder')
    parser.add_argument('-c2', '--category2', default='', help='2nd folder')
    args = parser.parse_args()
    image_groups = { args.category1 + 'train' : [], args.category1 + 'test': [], \
         args.category2 + 'train': [], args.category2 + 'test': []}
    return args.directory, [args.category1, args.category2], image_groups

def split_images(directory, categories, image_groups):
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

    return image_groups

def copy_images_to_new_directories(directory, categories, image_groups):
    # create new images folders
    for folder in image_groups:
        if not os.path.exists(os.path.join(directory, folder)):
            os.mkdir(os.path.join(directory, folder))
    for (group, img_list) in image_groups.items():
        for img in img_list:
            current_loc = os.path.join(directory, group.replace('test', '').replace('train',''), img)
            new_loc = os.path.join(directory, group, img)
            shutil.copyfile(current_loc, new_loc)
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
