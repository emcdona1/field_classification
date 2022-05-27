import os
from models import CNNModel
from labeled_images.labeledimages import LabeledImages
import numpy as np
from data_visualization.visualizationgenerator import VisualizationGenerator
from tensorflow import keras
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
        fold_groups = zip(images.training_image_set, images.validation_image_set)
        for index, (training_set, validation_set) in enumerate(fold_groups):
            self.curr_index = index
            self.train_model(training_set, validation_set)
            keras.models.save_model(self.architecture.model,
                                    os.path.join(self.folder_name, f'CNN_{self.curr_index + 1}.model'))
            self.validate_model_at_epoch_end(images, validation_set)

    def train_model(self, training_set: tf.data.Dataset, validation_set: tf.data.Dataset) -> None:
        self.architecture.reset_model()
        print(f'Training model for fold {self.curr_index + 1} of {self.n_folds}.')
        new_history = self.architecture.model.fit(training_set,
                                                  validation_data=validation_set,
                                                  batch_size=self.batch_size,
                                                  epochs=self.epochs,
                                                  verbose=2)
        self.history.append(new_history)

    def validate_model_at_epoch_end(self, images: LabeledImages, validation_set: tf.data.Dataset) -> None:
        validation_labels = np.array([])
        validation_predicted_probability = self.architecture.model.predict(validation_set)[:, 1]  # ,0]
        for batch in validation_set.as_numpy_iterator():
            validation_labels = np.concatenate((validation_labels, batch[1]))
        validation_labels = validation_labels.astype(np.int32)
        predictions = self.architecture.model.predict(validation_set)
        current_class = 0

        print('Classes: ', len(images.class_labels))

        self.charts.update(self.curr_index + 1, validation_labels, validation_predicted_probability,
                           self.history[self.curr_index], images.class_labels, predictions, current_class)
