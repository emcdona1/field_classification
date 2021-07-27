import sys

import Augmentor


def main(train_path, b1, val_path, b2):
    # train_path = r"C:\Users\think\PycharmProjects\field_classification\5_leaf_catagories\model12_2and3_aug1_ab\training_images"
    tp = Augmentor.Pipeline(train_path)

    tp.rotate(probability=1.0, max_left_rotation=15, max_right_rotation=15)
    tp.flip_random(probability=1.0)

    # tp.rotate(probability=.5, max_left_rotation=15, max_right_rotation=15)
    # tp.rotate180(probability=.5)
    # tp.flip_random(probability=.5)
    # tp.shear(probability=.5, max_shear_left=10, max_shear_right=10)
    # tp.skew(probability=.5)

    tp.sample(int(b1))

    # val_path = r'C:\Users\think\PycharmProjects\field_classification\5_leaf_catagories\model12_2and3_aug1_ab\testing_images'
    vp = Augmentor.Pipeline(val_path)

    vp.rotate(probability=1.0, max_left_rotation=15, max_right_rotation=15)
    vp.flip_random(probability=1.0)

    # vp.rotate(probability=.5, max_left_rotation=15, max_right_rotation=15)
    # vp.rotate180(probability=.5)
    # vp.flip_random(probability=.5)
    # vp.shear(probability=.5, max_shear_left=10, max_shear_right=10)
    # vp.skew(probability=.5)

    vp.sample(int(b2))


if __name__ == '__main__':
    assert len(sys.argv) >= 5, 'Please specify at least one image folder.'
    train_path = sys.argv[1]
    b1 = sys.argv[2]
    val_path = sys.argv[3]
    b2 = sys.argv[4]
    main(train_path, b1, val_path, b2)
