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

    cnn_model_trainer.train_and_save_all_models(images)
    cnn_model_trainer.charts.finalize()

    print('class 1: ' + images.class_labels[0] + ', class 2: ' + images.class_labels[1])
    timer.stop()
    timer.print_results()


def program_setup() -> (LabeledImages, ModelTrainer):
    cnn_arguments = CNNArguments()

    images = load_image_sets(cnn_arguments)
    trainer = initialize_model_trainer(cnn_arguments, images)
    return images, trainer


def load_image_sets(user_arguments: CNNArguments) -> LabeledImages:
    # load from hex-->ascii = import codecs;codecs.decode('<folder name>', 'hex')
    images = LabeledImages(SEED)
    # Option 1: load from filesystem
    images.load_images_from_folders(user_arguments.image_folders, user_arguments.color_mode,
                                    user_arguments.class_labels)

    # Option 2: load 2 classes of the CIFAR-10 data set
    # images.load_cifar_images()

    return images


def initialize_model_trainer(user_arguments: CNNArguments, images: LabeledImages) -> (ModelTrainer):
    n_folds = user_arguments.n_folds
    architecture = SmithsonianModel(SEED, user_arguments.lr, images.image_dimension, images.color_mode)
    trainer = ModelTrainer(user_arguments.n_epochs, user_arguments.batch_size, n_folds, architecture, SEED)
    return trainer


if __name__ == '__main__':
    # set up random seeds
    SEED = 1
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.compat.v1.random.set_random_seed(SEED)  # todo: remove this?
    random.seed(SEED)

    main()
