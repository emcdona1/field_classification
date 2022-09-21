import os
import sys
import Augmentor
from pathlib import Path


def main(image_source_folder: Path):

    image_folders = [Path(image_source_folder, f) for f in os.listdir(image_source_folder)]
    for folder, batch_size in image_folders:
        augmentation_pipeline = Augmentor.Pipeline(source_directory=folder)  #, output_directory=)

        # Augmentation option #1
        augmentation_pipeline.rotate(probability=1.0, max_left_rotation=15, max_right_rotation=15)
        augmentation_pipeline.flip_random(probability=1.0)

        # Augmentation option #2
        # augmentation_pipeline.rotate(probability=.5, max_left_rotation=15, max_right_rotation=15)
        # augmentation_pipeline.rotate180(probability=.5)
        # augmentation_pipeline.flip_random(probability=.5)
        # augmentation_pipeline.shear(probability=.5, max_shear_left=10, max_shear_right=10)
        # augmentation_pipeline.skew(probability=.5)

        # Augmentation option #3
        # augmentation_pipeline.rotate(probability=.5, max_left_rotation=20, max_right_rotation=20)
        # augmentation_pipeline.rotate180(probability=.5)
        # augmentation_pipeline.flip_random(probability=1)

        # Augmentation option #4
        # augmentation_pipeline.rotate(probability=1, max_right_rotation=20, max_left_rotation=20)
        # augmentation_pipeline.flip_random(probability=1)
        # augmentation_pipeline.crop_random(probability=1, percentage_area=.9)
        # augmentation_pipeline.shear(probability=1, max_shear_left=20, max_shear_right=20)
        # augmentation_pipeline.skew(probability=1)

        augmentation_pipeline.sample(batch_size)


if __name__ == '__main__':
    assert len(sys.argv) == 2, 'Please specify 1 argument: 1) a folder that contains folders of TRAINING images.  ' +\
                               '(Note: you should not augment testing images).'
    training_image_folder = Path(sys.argv[1])
    assert training_image_folder.exists() and training_image_folder.is_dir(), \
        f'Not a valid folder path: {training_image_folder}'
    batch_size_1 = 64  # int(sys.argv[2])

    validation_image_folder = Path(sys.argv[3])
    assert validation_image_folder.exists() and validation_image_folder.is_dir(), \
        f'Not a valid folder path: {validation_image_folder}'
    batch_size_2 = int(sys.argv[4])

    main(training_image_folder, batch_size_1, validation_image_folder, batch_size_2)
