import random
import numpy as np
import tensorflow as tf
import matplotlib
from labeled_images.labeledimages import LabeledImages
from models.smithsonian import SmithsonianModel
from model_training import ModelTrainer
from utilities.timer import Timer
from modeltrainingarguments import ModelTrainingArguments

matplotlib.use('Agg')  # required when running on server


def main() -> None:
    timer = Timer('Model training')

    cnn_arguments = ModelTrainingArguments()
    new_images = LabeledImages(SEED)
    new_images.load_images_from_folders(cnn_arguments.training_image_folder, cnn_arguments.image_size,
                                        cnn_arguments.color_mode, shuffle=True, n_folds=cnn_arguments.n_folds)

    architecture = SmithsonianModel(SEED, cnn_arguments.lr, cnn_arguments.image_size, cnn_arguments.color_mode)
    trainer = ModelTrainer(cnn_arguments.n_epochs, cnn_arguments.batch_size, cnn_arguments.n_folds, architecture, SEED)
    trainer.train_and_save_all_models(new_images)
    trainer.charts.finalize()

    print('class 1: ' + new_images.class_labels[0] + ', class 2: ' + new_images.class_labels[1])
    timer.stop()
    timer.print_results()


if __name__ == '__main__':
    # set up random seeds
    SEED = 1
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)

    main()
