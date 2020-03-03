import argparse
import os


def initialize_argparse(parser) -> argparse.Namespace:
    # image arguments
    parser.add_argument('c1', help='Directory name containing images in class 1.')
    parser.add_argument('c2', help='Directory name containing images in class 2.')
    parser.add_argument('-s', '--img_size', type=int, default=256,
                        help='Image dimension in pixels (must be square).')
    color_mode_group = parser.add_mutually_exclusive_group()
    color_mode_group.add_argument('-color', action='store_true', help='(default) Images are in RGB color mode.')
    color_mode_group.add_argument('-bw', action='store_true', help='Images are in grayscale color mode.')

    # model creation argument
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='Learning rate for training.')

    # training run arguments
    parser.add_argument('-f', '--n_folds', type=int, default=10,
                        help='Number of folds (minimum 2) for cross validation.')
    parser.add_argument('-e', '--n_epochs', type=int, default=25, help='Number of epochs.')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size for training.')

    return parser.parse_args()


def validate_image_folders(args: argparse.Namespace) -> tuple:
    if not os.path.isdir(args.c1):
        raise NotADirectoryError('%s is not a valid directory path.' % args.c1)
    if not os.path.isdir(args.c2):
        raise NotADirectoryError('%s is not a valid directory path.' % args.c2)
    image_folders = (args.c1, args.c2)

    c1 = args.c1.strip(os.path.sep)
    c2 = args.c2.strip(os.path.sep)
    c1 = c1.split(os.path.sep)[c1.count(os.path.sep)]
    c2 = c2.split(os.path.sep)[c2.count(os.path.sep)]
    class_labels = (c1, c2)

    return image_folders, class_labels


def validate_image_size(args: argparse.Namespace) -> int:
    img_size = args.img_size
    if not img_size >= 4:
        raise ValueError('%i is not a valid image dimension (in pixels). Must be >= 4.' % img_size)
    return img_size


def validate_learning_rate(args: argparse.Namespace) -> float:
    lr = args.learning_rate
    if not 0 < lr <= 1:
        raise ValueError('%f.6 is not a valid learning rate. Must be in range 0 (exclusive) to 1 (inclusive).' % lr)
    return lr


def validate_n_folds(args: argparse.Namespace) -> int:
    n_folds = args.n_folds
    if not n_folds >= 2:
        raise ValueError('%i is not a valid number of folds. Must be >= 2.' % n_folds)

    return n_folds


def validate_n_epochs(args: argparse.Namespace) -> int:
    n_epochs = args.n_epochs
    # if not n_epochs >= 10:
    #     raise ValueError('%i is not a valid number of epochs. Must be >= 10.)' % n_epochs)
    return n_epochs


def validate_batch_size(args: argparse.Namespace) -> int:
    batch_size = args.batch_size
    if not batch_size >= 2:
        raise ValueError('%i is not a valid batch size. Must be >= 2.' % batch_size)
    return batch_size
