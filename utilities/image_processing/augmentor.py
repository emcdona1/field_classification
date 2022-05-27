import sys
import Augmentor
from pathlib import Path

    # aug3
    # tp.rotate(probability=.5, max_left_rotation=20, max_right_rotation=20)
    # tp.rotate180(probability=.5)
    # tp.flip_random(probability=1)

    # aug4
    # tp.rotate(probability=1, max_right_rotation=20, max_left_rotation=20)
    # tp.flip_random(probability=1)
    # tp.crop_random(probability=1, percentage_area=.9)
    # tp.shear(probability=1, max_shear_left=20, max_shear_right=20)
    # tp.skew(probability=1)

    tp.sample(int(b1))

    vp = Augmentor.Pipeline(val_path)

    # aug1
    vp.rotate(probability=1.0, max_left_rotation=15, max_right_rotation=15)
    vp.flip_random(probability=1.0)

    # aug2
    # vp.rotate(probability=.5, max_left_rotation=15, max_right_rotation=15)
    # vp.rotate180(probability=.5)
    # vp.flip_random(probability=.5)
    # vp.shear(probability=.5, max_shear_left=10, max_shear_right=10)
    # vp.skew(probability=.5)

    # aug3
    # vp.rotate(probability=.5, max_left_rotation=20, max_right_rotation=20)
    # vp.rotate180(probability=.5)
    # vp.flip_random(probability=1)

    # aug4
    # vp.rotate(probability=1, max_right_rotation=20, max_left_rotation=20)
    # vp.flip_random(probability=1)
    # vp.crop_random(probability=1, percentage_area=.9)
    # vp.shear(probability=1, max_shear_left=20, max_shear_right=20)
    # vp.skew(probability=1)

    vp.sample(int(b2))


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
