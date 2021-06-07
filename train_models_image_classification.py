import random
import numpy as np
import tensorflow as tf
import matplotlib
from labeled_images.labeledimages import LabeledImages, NewLabeledImages
from models.smithsonian import SmithsonianModel
from model_training import ModelTrainer
from utilities.timer import Timer
from cnnarguments import CNNArguments

matplotlib.use('Agg')  # required when running on server


def main() -> None:
    timer = Timer('Model training')

    cnn_arguments = CNNArguments()
    new_images = NewLabeledImages(SEED)
    new_images.load_images_from_folders(cnn_arguments.training_image_folder, cnn_arguments.image_size,
                                        cnn_arguments.color_mode, shuffle=True, n_folds=cnn_arguments.n_folds)
    # todo: get rid of comments below
    # images = LabeledImages()
    # Option 1: load from filesystem
    # images.load_images_from_folders(cnn_arguments.image_folders, cnn_arguments.color_mode, cnn_arguments.class_labels)
    # Option 2: load 2 classes of the CIFAR-10 data set
    # images.load_cifar_images()

    architecture = SmithsonianModel(SEED, cnn_arguments.lr, cnn_arguments.image_size, cnn_arguments.color_mode)
    trainer = ModelTrainer(cnn_arguments.n_epochs, cnn_arguments.batch_size, cnn_arguments.n_folds, architecture, SEED)
    trainer.new_train_and_save_all_models(new_images)
    # trainer.train_and_save_all_models(images)
    trainer.charts.finalize()

    print('class 1: ' + new_images.class_labels[0] + ', class 2: ' + new_images.class_labels[1])
    timer.stop()
    timer.print_results()


if __name__ == '__main__':
    # set up random seeds
    SEED = 1
    np.random.seed(SEED)
    tf.compat.v1.random.set_random_seed(SEED)
    random.seed(SEED)

    main()
