import argparse
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import matplotlib
from labeled_images.labeledimages import LabeledImages
from models.smithsonian import SmithsonianModel
from data_and_visualization_io import Charts
from model_training import ModelTrainer
from timer import Timer
from cnnarguments import CNNArguments

matplotlib.use('Agg')  # required when running on server


def main() -> None:
    timer = Timer('Model training')
    class_labels, images, trainer, n_folds, charts = setup()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for index, (training_idx_list, validation_idx_list) in enumerate(skf.split(images.features, images.labels)):
        trainer.set_up_new_model(images, training_idx_list, validation_idx_list, index)
        trainer.train_new_model()
        trainer.save_model()
        trainer.validate_model(charts, class_labels)
    charts.finalize()

    print('class 1: ' + class_labels[0] + ', class 2: ' + class_labels[1])
    timer.stop()
    timer.results()


def setup():
    parser = argparse.ArgumentParser(
        'Create and train CNNs for binary classification of images, using cross-fold validation.')
    user_arguments = CNNArguments(parser)
    image_folders, class_labels = user_arguments.image_folders_and_class_labels()
    n_folds = user_arguments.n_folds()
    images = LabeledImages(image_folders, user_arguments.color_mode(), SEED)
    architecture = SmithsonianModel(SEED, user_arguments.learning_rate(), images.size)
    trainer = ModelTrainer(user_arguments.n_epochs(), user_arguments.batch_size(), n_folds, architecture)
    charts = Charts(n_folds)

    return class_labels, images, trainer, n_folds, charts


if __name__ == '__main__':
    # set up random seeds
    SEED = 1
    np.random.seed(SEED)
    tf.compat.v1.random.set_random_seed(SEED)
    random.seed(SEED)

    main()
