import os
from models.smithsonian import SmithsonianModel
from labeled_images.labeledimages import LabeledImages
import numpy as np
from data_visualization.visualizationgenerator import VisualizationGenerator
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import save_model
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

        for index, (training_idx_list, validation_idx_list) in training_and_validation_groups:
            self.curr_fold = index + 1
            self.train_model(images, training_idx_list, validation_idx_list)
            self.save_model()
            self.validate_model(images.class_labels)

    def generate_image_splits(self, images: LabeledImages) -> enumerate:
        if self.n_folds <= 1:
            print('Training without cross-fold validation.')
            # 90% training, 10% validation
            training_idx_list = np.array(range(int(images.no_of_images * 0.9)))
            validation_idx_list = np.array(range(len(training_idx_list), images.no_of_images))
            training_and_validation_groups = enumerate([(training_idx_list, validation_idx_list)])
        else:
            print('Training with %i-fold validation.' % self.n_folds)
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            training_and_validation_groups = enumerate(skf.split(images.features, images.labels))

        return training_and_validation_groups

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

    def save_model(self):
        save_model(self.architecture.model, os.path.join(self.folder_name, 'CNN_' + str(self.curr_fold) + '.model'))

    def validate_model(self, class_labels: Union[Tuple[str, ...], List[str, ...]]) -> None:
        # validation_predicted_probability = self.architecture.model.predict_proba(self.validation_set[0])[:, 1]
        validation_predicted_probability = self.architecture.model.predict(self.validation_set[0])[:, 1]
        self.charts.update(self.history, self.curr_fold, self.validation_set[1],
                           validation_predicted_probability, class_labels)
