import os
from models.smithsonian import SmithsonianModel
from labeled_images.labeledimages import LabeledImages
import numpy as np
from data_and_visualization_io import Charts


class ModelTrainer:
    def __init__(self, epochs: int, batch_size: int, n_folds: int, architecture: SmithsonianModel):
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

    def train_a_model(self, images: LabeledImages, training_idx_list: np.ndarray, validation_idx_list: np.ndarray,
                      curr_fold: int) -> None:
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

    def save_model(self):
        self.architecture.model.save(os.path.join(self.folder_name, 'CNN_' + str(self.curr_fold) + '.model'))

    def validate_model(self, charts: Charts, class_labels: tuple) -> None:
        validation_predicted_probability = self.architecture.model.predict_proba(self.validation_set[0])[:, 1]
        charts.update(self.history, self.curr_fold, self.validation_set[1],
                      validation_predicted_probability, class_labels)
