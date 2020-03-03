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
import warnings
from validation import Validation

matplotlib.use('Agg')  # required when running on server


def main() -> None:
    timer = Timer('Model training')
    class_labels, images, architecture, trainer, n_folds, charts = setup()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for index, (training_idx_list, validation_idx_list) in enumerate(skf.split(images.features, images.labels)):
        # set up this model run
        architecture.reset_model()
        training_set = images.subset(training_idx_list)
        validation_set = images.subset(validation_idx_list)

        # train model
        history = trainer.train_new_model_and_save(architecture, training_set, validation_set, index, n_folds)

        # validate newly created model
        validation_predicted_probability = architecture.model.predict_proba(validation_set[0])[:, 1]
        charts.update(history, index, validation_set[1], validation_predicted_probability, class_labels)

    finalize(charts, class_labels, timer)


def setup():
    image_folders, class_labels, img_size, color_mode, lr, n_folds, n_epochs, batch_size = get_arguments()

    trainer = ModelTrainer(n_epochs, batch_size)

    # Load in images and shuffle order
    images = LabeledImages(image_folders, color_mode, SEED)
    architecture = SmithsonianModel(SEED, lr)

    charts = Charts(n_folds)

    return class_labels, images, architecture, trainer, n_folds, charts


def get_arguments():
    parser = argparse.ArgumentParser(
        'Create and train CNNs for binary classification of images, using cross-fold validation.')
    val = Validation(parser)
    image_folders, class_labels = val.validate_image_folders()
    img_size = val.validate_image_size()
    lr = val.validate_learning_rate()
    color_mode = val.validate_color_mode()
    n_folds = val.validate_n_folds()
    n_epochs = val.validate_n_epochs()
    batch_size = val.validate_batch_size()
    return image_folders, class_labels, img_size, color_mode, lr, n_folds, n_epochs, batch_size


def finalize(charts, class_labels, timer):
    charts.finalize()
    # end
    print('class 1: ' + class_labels[0] + ', class 2: ' + class_labels[1])
    timer.stop()
    timer.results()


if __name__ == '__main__':
    # set up random seeds
    SEED = 1
    np.random.seed(SEED)
    tf.compat.v1.random.set_random_seed(SEED)
    random.seed(SEED)

    warnings.filterwarnings('ignore', category=DeprecationWarning)

    main()
