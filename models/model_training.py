import os
from models import CNNModel
from labeled_images.labeledimages import LabeledImages
import numpy as np
from data_visualization.visualizationgenerator import VisualizationGenerator
from tensorflow import keras
from typing import Union, Tuple, List
import tensorflow as tf


class ModelTrainer:
    def __init__(self, epochs: int, batch_size: int, n_folds: int, architecture: CNNModel, seed: int):
        self.epochs = epochs
        self.batch_size = batch_size
        self.folder_name = 'saved_models'
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
        self.architecture = architecture
        self.n_folds = n_folds
        self.curr_index = 1
        self.history = list()
        self.seed: int = seed
        self.charts = VisualizationGenerator(self.n_folds)

    def train_and_save_all_models(self, images: LabeledImages):
        # training_groups, validation_groups = self.generate_image_splits(images)
        fold_groups = zip(images.training_image_set, images.validation_image_set)
        for index, (training_set, validation_set) in enumerate(fold_groups):
            self.curr_index = index
            self.train_model(training_set, validation_set)
            keras.models.save_model(self.architecture.model,
                                    os.path.join(self.folder_name, f'CNN_{self.curr_index + 1}.model'))
            self.validate_model_at_epoch_end(images.class_labels, validation_set)

    def train_model(self, training_set: tf.data.Dataset, validation_set: tf.data.Dataset) -> None:
        self.architecture.reset_model()
        print(f'Training model for fold {self.curr_index + 1} of {self.n_folds}.')
        new_history = self.architecture.model.fit(training_set,
                                                  validation_data=validation_set,
                                                  batch_size=self.batch_size,
                                                  epochs=self.epochs,
                                                  verbose=2)
        self.history.append(new_history)

    def validate_model_at_epoch_end(self, class_labels: Union[Tuple[str], List[str]],
                                    validation_set: tf.data.Dataset) -> None:
        # todo: possibly change this into model.evaluate?
        validation_predicted_probability = self.architecture.model.predict(validation_set)[:, 1]  # ,0]
        validation_labels = np.array([])
        for batch in validation_set.as_numpy_iterator():
            validation_labels = np.concatenate((validation_labels, batch[1]))
        validation_labels = validation_labels.astype(np.int32)
        self.charts.update(self.history[self.curr_index], self.curr_index + 1, validation_labels,
                           validation_predicted_probability, class_labels)
