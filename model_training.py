import os
from models.smithsonian import SmithsonianModel
from labeled_images.labeledimages import LabeledImages
import numpy as np
from data_and_visualization_io import Charts
from sklearn.model_selection import StratifiedKFold


class ModelTrainer:
    def __init__(self, epochs: int, batch_size: int, n_folds: int, architecture: SmithsonianModel, seed: int):
        self.epochs = epochs
        self.batch_size = batch_size
        self.folder_name = 'saved_models'
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
        self.architecture = architecture
        self.n_folds = n_folds
        self.curr_fold = 0
        self.training_set = None
        self.validation_set = None
        self.history = None
        self.seed: int = seed
        self.charts = Charts(self.n_folds)

    def train_all_models(self, images: LabeledImages):
        if self.n_folds <= 1:
            print('Training without cross-fold validation.')
            # 90% training, 10% validation
            training_idx_list = np.array(range(int(images.n_images * 0.9)))
            validation_idx_list = np.array(range(len(training_idx_list), images.n_images))
            self.train_and_validate_a_model(images, training_idx_list, validation_idx_list, 0)
        else:
            print('Training with %i-fold validation.' % self.n_folds)
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            for index, (training_idx_list, validation_idx_list) in enumerate(skf.split(images.features, images.labels)):
                self.train_and_validate_a_model(images, training_idx_list, validation_idx_list, index)
        self.charts.finalize()

    def train_and_validate_a_model(self, images: LabeledImages, training_idx_list: np.ndarray,
                                   validation_idx_list: np.ndarray,
                                   curr_fold: int):
        self.architecture.reset_model()
        self.history = None
        self.training_set = images.subset(training_idx_list)
        self.validation_set = images.subset(validation_idx_list)
        self.curr_fold: int = curr_fold + 1

        print('Training model for fold %i of %i.' % (self.curr_fold, self.n_folds))
        # es_callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', \
        #        mode='min', min_delta = 0.05, patience = 20, restore_best_weights = True)
        self.history = self.architecture.model.fit(self.training_set[0], self.training_set[1],
                                                   batch_size=self.batch_size, epochs=self.epochs,
                                                   #        callbacks = [es_callback], \
                                                   validation_data=self.validation_set, verbose=2)
        self.save_model()
        self.validate_model(images.class_labels)

    def save_model(self):
        self.architecture.model.save(os.path.join(self.folder_name, 'CNN_' + str(self.curr_fold) + '.model'))

    def validate_model(self, class_labels: tuple) -> None:
        validation_predicted_probability = self.architecture.model.predict_proba(self.validation_set[0])[:, 1]
        self.charts.update(self.history, self.curr_fold, self.validation_set[1],
                           validation_predicted_probability, class_labels)
