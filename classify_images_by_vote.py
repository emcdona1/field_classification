THRESHOLD = 0.25

def process_input_arguments():
    parser = argparse.ArgumentParser('Import a model and classify images.')
    parser.add_argument('-d', '--directory', default='images', help='Folder holding image folders')	
    parser.add_argument('-c1', '--category1', help='Folder of class 1')
    parser.add_argument('-c2', '--category2', help='Folder of class 2')
    parser.add_argument('-s', '--img_size', default=256, help='Image dimension in pixels')
    parser.add_argument('-m', '--models', help='Directory of models to use')
    args = parser.parse_args()

    img_directory = args.directory
    folders = [args.category1, args.category2]
    img_size = args.img_size
    model_directory = args.models

    return img_directory, folders, img_size, model_directory

if __name__ == '__main__':
    # Start execution and parse arguments
    start_time = time.time()
    img_directory, folders, img_size, model_directory = process_input_arguments()

    # Import images
    pixel_values, actual_class, img_filenames = import_images(img_directory, folders, img_size)
    print('Images imported.')

    # for each model in model_directory:
    #   load model
    #   classify each image (make a 5-col chart)
    # for each row in chart, vote (simple majority) and give each *image* a final classification
    # first output: image name, final classification, true classification, tp/fn/fp/tn, then each classification
    
    # then, aggregate by *plant* - the first part of the filename up to the underscore
    # simple majority to classify each plant
    # second output: plant specimen name, final classification, true classification, tp/fn/fp/tn


    # Finish execution
    end_time = time.time()
    print('Completed in %.1f seconds' % (end_time - start_time))