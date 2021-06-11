import os
from models.smithsonian import SmithsonianModel
from labeled_images.labeledimages import LabeledImages
import numpy as np
from data_visualization.visualizationgenerator import VisualizationGenerator
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from typing import Union, Tuple, List
import tensorflow as tf


class ModelTrainer:
    def __init__(self, epochs: int, batch_size: int, n_folds: int, architecture: SmithsonianModel, seed: int):
        self.epochs = epochs
        self.batch_size = batch_size
        self.folder_name = 'saved_models'
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
        self.architecture = architecture
        self.n_folds = n_folds
        self.curr_fold = 1
        self.history = None
        self.seed: int = seed
        self.charts = VisualizationGenerator(self.n_folds)

    def train_and_save_all_models(self, images: LabeledImages):
        training_groups, validation_groups = self.generate_image_splits(images)
        for index, (training_set, validation_set) in enumerate(zip(training_groups, validation_groups)):
            self.curr_fold = index + 1
            self.train_model(training_set, validation_set)
            keras.models.save_model(self.architecture.model,
                                    os.path.join(self.folder_name, 'CNN_%i.model' % self.curr_fold))
            self.validate_model(images.class_labels, validation_set)

    def generate_image_splits(self, images: LabeledImages) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        if self.n_folds == 1:
            print('Training without cross-fold validation.')
        else:
            print('Training with %i cross-fold validation.' % images.n_folds)
            # TODO: Remove the warning below once no longer relevant
            print('WARNING: Cross-fold validation has not been implemented!')
        training_set = images.training_image_set
        validation_set = images.validation_image_set
        return training_set, validation_set

    def train_model(self, training_set: tf.data.Dataset, validation_set: tf.data.Dataset) -> None:
        self.architecture.reset_model()
        self.history = None
        print('Training model for fold %i of %i.' % (self.curr_fold, self.n_folds))
        self.history = self.architecture.model.fit(training_set,
                                                   batch_size=self.batch_size, epochs=self.epochs,
                                                   #        callbacks = [es_callback],
                                                   validation_data=validation_set, verbose=2)

    def validate_model(self, class_labels: Union[Tuple[str], List[str]], validation_set: tf.data.Dataset) -> None:
        validation_predicted_probability = self.architecture.model.predict(validation_set)[:, 1]  # ,0]
        validation_labels = np.array([])
        for batch in validation_set.as_numpy_iterator():
            validation_labels = np.concatenate((validation_labels, batch[1]))
        validation_labels = validation_labels.astype(np.int32)
        self.charts.update(self.history, self.curr_fold, validation_labels,
                           validation_predicted_probability, class_labels)
