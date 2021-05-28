import os
from models.smithsonian import SmithsonianModel
from labeled_images.labeledimages import LabeledImages, NewLabeledImages
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

    def new_train_and_save_all_models(self, images: NewLabeledImages):
        training_and_validation_groups = self.new_generate_image_splits(images)
        for index, (training_set, validation_set) in enumerate(training_and_validation_groups):
            self.curr_fold = index + 1
            self.training_set = training_set
            self.validation_set = validation_set
            self.new_train_model()
            keras.models.save_model(self.architecture.model,
                                    os.path.join(self.folder_name, 'CNN_%i.model' % self.curr_fold))
            self.validate_model(images.class_labels)

    def train_and_save_all_models(self, images: LabeledImages):  # todo: remove
        training_and_validation_groups = self.generate_image_splits(images)

        for index, (training_idx_list, validation_idx_list) in enumerate(training_and_validation_groups):
            self.curr_fold = index + 1
            self.train_model(images, training_idx_list, validation_idx_list)
            keras.models.save_model(self.architecture.model, os.path.join(self.folder_name,
                                                                          'CNN_' + str(self.curr_fold) + '.model'))
            self.validate_model(images.class_labels)

    def new_generate_image_splits(self, images: NewLabeledImages):
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

    def generate_image_splits(self, images: LabeledImages):  # todo: remove
        if self.n_folds <= 1:
            print('Training without cross-fold validation.')
            # 90% training, 10% validation
            training_idx_list = np.array(range(int(images.img_count * 0.9)))
            validation_idx_list = np.array(range(len(training_idx_list), images.img_count))
            training_and_validation_groups = enumerate([(training_idx_list, validation_idx_list)])
        else:
            print('Training with %i-fold validation.' % self.n_folds)
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            training_and_validation_groups = skf.split(images.features, images.labels)

        return training_and_validation_groups

    def new_train_model(self):
        self.architecture.reset_model()
        self.history = None
        print('Training model for fold %i of %i.' % (self.curr_fold, self.n_folds))
        self.history = self.architecture.model.fit(self.training_set,
                                                   batch_size=self.batch_size, epochs=self.epochs,
                                                   #        callbacks = [es_callback],
                                                   validation_data=self.validation_set, verbose=2)

    def train_model(self, images: LabeledImages, training_idx_list: np.ndarray,
                    validation_idx_list: np.ndarray):
        self.architecture.reset_model()
        self.history = None
        self.training_set = images.subset(training_idx_list)
        self.validation_set = images.subset(validation_idx_list)

        print('Training model for fold %i of %i.' % (self.curr_fold, self.n_folds))
        # es_callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',
        #        mode='min', min_delta = 0.05, patience = 20, restore_best_weights = True)
        self.history = self.architecture.model.fit(self.training_set[0], self.training_set[1],
                                                   batch_size=self.batch_size, epochs=self.epochs,
                                                   #        callbacks = [es_callback],
                                                   validation_data=self.validation_set, verbose=2)

    def validate_model(self, class_labels: Union[Tuple[str], List[str]]) -> None:
        # validation_predicted_probability = self.architecture.model.predict_proba(self.validation_set[0])[:, 1]
        validation_predicted_probability = self.architecture.model.predict(self.validation_set)[:, 1]
        self.charts.update(self.history, self.curr_fold, self.validation_set[1],  # todo: how to extract the labels from the validation set?
                           validation_predicted_probability, class_labels)
