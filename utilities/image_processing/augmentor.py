import sys
import Augmentor


def main(train_path, b1, val_path, b2):
    tp = Augmentor.Pipeline(train_path)

    # aug1
    tp.rotate(probability=1.0, max_left_rotation=15, max_right_rotation=15)
    tp.flip_random(probability=1.0)

    # aug2
    # tp.rotate(probability=.5, max_left_rotation=15, max_right_rotation=15)
    # tp.rotate180(probability=.5)
    # tp.flip_random(probability=.5)
    # tp.shear(probability=.5, max_shear_left=10, max_shear_right=10)
    # tp.skew(probability=.5)

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
    assert len(sys.argv) >= 5, 'Please specify at least one image folder.'
    train_path = sys.argv[1]
    b1 = sys.argv[2]
    val_path = sys.argv[3]
    b2 = sys.argv[4]
    main(train_path, b1, val_path, b2)
