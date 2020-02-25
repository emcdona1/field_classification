import os


class ModelTrainer:
    def __init__(self, epochs: int, batch_size: int):
        self.epochs = epochs
        self.batch_size = batch_size

    def train_blank_model(self, model_architecture, training_set, validation_set, curr_fold, n_folds):
        print('Training model for fold %i of %i' % (curr_fold + 1, n_folds))
        # es_callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', \
        #        mode='min', min_delta = 0.05, patience = 20, restore_best_weights = True)
        history = model_architecture.model.fit(training_set[0], training_set[1],
                                               batch_size=self.batch_size, epochs=self.epochs,
                                               #        callbacks = [es_callback], \
                                               validation_data=(validation_set[0], validation_set[1]), verbose=2)
        model_architecture.model.save(os.path.join('saved_models', 'CNN_%i.model' % curr_fold + 1))

        return history
