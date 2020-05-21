import argparse
import random
import numpy as np
import tensorflow as tf
import matplotlib
from labeled_images.labeledimages import LabeledImages
from models.smithsonian import SmithsonianModel
from model_training import ModelTrainer
from utilities.timer import Timer
from cnnarguments import CNNArguments

matplotlib.use('Agg')  # required when running on server


def main() -> None:
    timer = Timer('Model training')
    images, cnn_model_trainer = program_setup()

    cnn_model_trainer.train_all_models(images)

    print('class 1: ' + images.class_labels[0] + ', class 2: ' + images.class_labels[1])
    timer.stop()
    timer.print_results()


def program_setup() -> (LabeledImages, ModelTrainer):
    parser = argparse.ArgumentParser(
        'Create and train CNNs for binary classification of images, using cross-fold validation.')
    user_arguments = CNNArguments(parser)

    images = load_image_sets(user_arguments)
    trainer = initialize_model_trainer(user_arguments, images)
    return images, trainer


def load_image_sets(user_arguments: CNNArguments) -> LabeledImages:
    images = LabeledImages(SEED)
    # Option 1: load from filesystem
    image_folders, class_labels = user_arguments.image_folders_and_class_labels()
    color_mode = user_arguments.color_mode()
    images.load_images_from_folders(image_folders, color_mode, class_labels)

    # Option 2: load 2 classes of the CIFAR-10 data set
    # images.load_cifar_images()

    return images


def initialize_model_trainer(user_arguments: CNNArguments, images: LabeledImages) -> (ModelTrainer):
    n_folds = user_arguments.n_folds()
    architecture = SmithsonianModel(SEED, user_arguments.learning_rate(), images.img_dim, images.color_mode)
    trainer = ModelTrainer(user_arguments.n_epochs(), user_arguments.batch_size(), n_folds, architecture, SEED)
    return trainer


if __name__ == '__main__':
    # set up random seeds
    SEED = 1
    np.random.seed(SEED)
    tf.compat.v1.random.set_random_seed(SEED)
    random.seed(SEED)

    main()
