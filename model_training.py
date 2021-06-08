import os
from models.smithsonian import SmithsonianModel
from labeled_images.labeledimages import LabeledImages
import numpy as np
from data_visualization.visualizationgenerator import VisualizationGenerator
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from typing import Union, Tuple, List


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
        self.training_set = None
        self.validation_set = None
        self.history = None
        self.seed: int = seed
        self.charts = VisualizationGenerator(self.n_folds)

    def train_and_save_all_models(self, images: LabeledImages):
        training_and_validation_groups = self.generate_image_splits(images)
        for index, (training_set, validation_set) in enumerate(training_and_validation_groups):
            self.curr_fold = index + 1
            self.training_set = training_set
            self.validation_set = validation_set
            self.train_model()
            keras.models.save_model(self.architecture.model,
                                    os.path.join(self.folder_name, 'CNN_%i.model' % self.curr_fold))
            self.validate_model(images.class_labels)

    def generate_image_splits(self, images: LabeledImages):
        if self.n_folds <= 1:
            print('Training without cross-fold validation.')
            # training_idx_list = np.array(range(int(images.img_count * 0.9)))
            # validation_idx_list = np.array(range(len(training_idx_list), images.img_count))
            # training_and_validation_groups = [(training_idx_list, validation_idx_list)]
            # 90% training, 10% validation
            training_set = images.training_image_set
            validation_set = images.validation_image_set
            training_and_validation_groups = [(training_set, validation_set)]
        else:  # todo: figure out how to do StratifiedKFold with tf.data.Dataset
            print('Training with %i-fold validation.' % self.n_folds)
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            training_and_validation_groups = skf.split('features', 'labels')
        return training_and_validation_groups

    def train_model(self):
        self.architecture.reset_model()
        self.history = None
        print('Training model for fold %i of %i.' % (self.curr_fold, self.n_folds))
        self.history = self.architecture.model.fit(self.training_set,
                                                   batch_size=self.batch_size, epochs=self.epochs,
                                                   #        callbacks = [es_callback],
                                                   validation_data=self.validation_set, verbose=2)

    def validate_model(self, class_labels: Union[Tuple[str], List[str]]) -> None:
        # validation_predicted_probability = self.architecture.model.predict_proba(self.validation_set[0])[:, 1]
        validation_predicted_probability = self.architecture.model.predict(self.validation_set)[:, 1]
        validation_labels = np.array([])
        for batch in self.validation_set.as_numpy_iterator():
            validation_labels = np.concatenate((validation_labels, batch[1]))
        validation_labels = validation_labels.astype(np.int32)
        self.charts.update(self.history, self.curr_fold, validation_labels,  # self.validation_set[1],  # todo: how to extract the labels from the validation set?
                           validation_predicted_probability, class_labels)
