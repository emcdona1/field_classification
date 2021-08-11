import random
from pathlib import Path
import numpy as np
import tensorflow as tf
from labeled_images import ColorMode
from labeled_images.labeledimages import LabeledImages
from models.rnn_ctc import RnnCtc
from models.model_training import CtcModelTrainer
from utilities.timer import Timer
from models.modeltrainingarguments import ModelTrainingArguments


def main() -> None:
    timer = Timer('Model training')

    args = ModelTrainingArguments()
    training_image_folder = args.training_image_folder
    image_size = args.image_size
    color_mode = args.color_mode
    n_folds = args.n_folds
    metadata = args.metadata
    lr = args.lr
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    # training_image_folder = Path('testing_image_sets/word_images')
    # image_size = (100, 400)
    # color_mode = ColorMode.grayscale
    # n_folds = 1
    # metadata = Path('testing_image_sets/word_images/words_metadata.csv')
    # lr = 0.01
    # n_epochs = 125
    # batch_size = 16

    images = LabeledImages(SEED)
    images.load_training_images(training_image_folder, image_size,
                                color_mode, shuffle=True, n_folds=n_folds,
                                metadata=metadata)

    architecture = RnnCtc(SEED, lr, image_size, color_mode)
    trainer = CtcModelTrainer(n_epochs, batch_size, n_folds, architecture, SEED)
    trainer.train_and_save_all_models(images)

    timer.stop()
    timer.print_results()


if __name__ == '__main__':
    # set up random seeds
    SEED = 1
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)

    main()
