import sys
import Augmentor
from pathlib import Path


def main(training_image_source_folder: Path, training_batch_size: int,
         validation_image_source_folder: Path, validation_batch_size: int):
    image_folders = [(training_image_source_folder, training_batch_size),
                     (validation_image_source_folder, validation_batch_size)]
    for folder, batch_size in image_folders:
        augmentation_pipeline = Augmentor.Pipeline(folder)

        # aug1
        augmentation_pipeline.rotate(probability=1.0, max_left_rotation=15, max_right_rotation=15)
        augmentation_pipeline.flip_random(probability=1.0)

        # aug2
        # augmentation_pipeline.rotate(probability=.5, max_left_rotation=15, max_right_rotation=15)
        # augmentation_pipeline.rotate180(probability=.5)
        # augmentation_pipeline.flip_random(probability=.5)
        # augmentation_pipeline.shear(probability=.5, max_shear_left=10, max_shear_right=10)
        # augmentation_pipeline.skew(probability=.5)

        # aug3
        # augmentation_pipeline.rotate(probability=.5, max_left_rotation=20, max_right_rotation=20)
        # augmentation_pipeline.rotate180(probability=.5)
        # augmentation_pipeline.flip_random(probability=1)

        # aug4
        # augmentation_pipeline.rotate(probability=1, max_right_rotation=20, max_left_rotation=20)
        # augmentation_pipeline.flip_random(probability=1)
        # augmentation_pipeline.crop_random(probability=1, percentage_area=.9)
        # augmentation_pipeline.shear(probability=1, max_shear_left=20, max_shear_right=20)
        # augmentation_pipeline.skew(probability=1)

        augmentation_pipeline.sample(batch_size)


if __name__ == '__main__':
    assert len(sys.argv) >= 5, 'Please specify at least 4 arguments: 1) training image folder and 2) batch size' +\
                               '3) validation image folder and 4) batch size.'
    training_image_folder = Path(sys.argv[1])
    assert training_image_folder.exists() and training_image_folder.is_dir(), \
        f'Not a valid folder path: {training_image_folder}'
    batch_size_1 = int(sys.argv[2])

    validation_image_folder = Path(sys.argv[3])
    assert validation_image_folder.exists() and validation_image_folder.is_dir(), \
        f'Not a valid folder path: {validation_image_folder}'
    batch_size_2 = int(sys.argv[4])

    main(training_image_folder, batch_size_1, validation_image_folder, batch_size_2)
